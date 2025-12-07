import click


"""
Regulatory Sequence Variant Prediction with TensorFlow 2.x
Optimized version with GPU/CPU auto-detection, batch processing, and modern TF2 compatibility
"""


def setup_tensorflow(use_gpu=True):
    """Configure TensorFlow 2.x for optimal performance on GPU or CPU."""
    import tensorflow as tf
    import os

    print(f"TensorFlow version: {tf.__version__}")

    if not use_gpu:
        # Explicitly disable GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled by user request")

    # Detect available devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    if gpus and use_gpu:
        print(f"Running on GPU: {len(gpus)} device(s) detected")
        for gpu in gpus:
            print(f"  - {gpu.name}")
            # Enable memory growth to avoid allocating all GPU memory at once
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"    Warning: {e}")

        # Enable mixed precision for faster inference on modern GPUs
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("  Mixed precision (FP16) enabled")
        except Exception as e:
            print(f"  Mixed precision not available: {e}")

        # Enable XLA JIT compilation for better performance
        tf.config.optimizer.set_jit(True)
        print("  XLA JIT compilation enabled")

        return True, tf.distribute.MirroredStrategy()
    else:
        if not use_gpu:
            print("Running on CPU (GPU disabled by user)")
        else:
            print("Running on CPU (no GPU detected)")

        # CPU optimizations
        # Let TensorFlow auto-detect optimal thread count
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)

        # Enable XLA for CPU as well
        tf.config.optimizer.set_jit(True)
        print("  CPU threading auto-configured")
        print("  XLA JIT compilation enabled")

        return False, None


def load_model_tf2(model_file, weights_file):
    """
    Load a TensorFlow/Keras model with TF2 compatibility.
    Handles both old Keras 2.x JSON models and modern TF2 models.
    """
    import tensorflow as tf
    import json

    print("Loading model architecture...")
    try:
        # Try loading as a complete saved model first (TF2 SavedModel format)
        model = tf.keras.models.load_model(model_file)
        print("  Loaded as TF2 SavedModel format")
        return model
    except (OSError, IOError, ValueError):
        pass

    # Try loading from JSON (old Keras 2.x format)
    try:
        with open(model_file, 'r') as f:
            model_json = f.read()

        # Parse the JSON to check the Keras version
        config = json.loads(model_json)
        keras_version = config.get('keras_version', 'unknown')
        print(f"  Detected Keras version: {keras_version}")

        # Use legacy model loading for old Keras 2.x models
        # TF 2.x includes backward compatibility for Keras 2.x through the legacy module
        try:
            # Try using the legacy save/load APIs
            if hasattr(tf.keras.saving, 'legacy'):
                from tensorflow.keras.saving.legacy import model_from_json
                print("  Using legacy model_from_json (TF2 backward compatibility)")
            else:
                from tensorflow.keras.models import model_from_json
                print("  Using model_from_json")

            model = model_from_json(model_json)
            print("  Loaded architecture from JSON")

        except Exception as json_error:
            # If direct JSON loading fails, try using the legacy deserialization
            print(f"  Direct JSON loading failed: {json_error}")
            print("  Attempting legacy deserialization...")

            # Use TF2's legacy serialization utilities
            from tensorflow.keras.utils import deserialize_keras_object
            from tensorflow.python.keras.saving import model_config as model_config_lib

            # Try to deserialize using TF2's internal methods
            model = model_config_lib.model_from_config(config['config'])
            print("  Loaded architecture using legacy deserialization")

        # Load weights
        if weights_file:
            print("Loading model weights...")
            model.load_weights(weights_file)
            print("  Weights loaded successfully")

        return model

    except Exception as e:
        print(f"  Error loading JSON model: {e}")
        print("\nTroubleshooting:")
        print("  - The model was saved with an older Keras version (2.3.0-tf)")
        print("  - TensorFlow 2.x has limited backward compatibility")
        print("  - Try re-saving the model in TensorFlow 2.x format if possible")
        raise RuntimeError(f"Could not load model from {model_file}") from e


# options
@click.command()
@click.option(
    "--variants",
    "variants_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Variant file to predict in VCF format.",
)
@click.option(
    "--model",
    "model_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Tensorflow model in json format or SavedModel directory.",
)
@click.option(
    "--weights",
    "weights_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Model weights in hdf5 format.",
)
@click.option(
    "--reference",
    "reference_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Reference sequence in FASTA format (indexed).",
)
@click.option(
    "--genome",
    "genome_file",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Genome file of the reference with lengths of contigs.",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(writable=True),
    default="/dev/stdout",
    help="Output file with predictions in tsv.gz format.",
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=32,
    help="Batch size for predictions (default: 32, increase for GPU)",
)
@click.option(
    "--use-gpu/--no-gpu",
    "use_gpu",
    default=True,
    help="Enable or disable GPU usage (default: enabled if available)",
)
def cli(
    variants_file, model_file, weights_file, reference_file, genome_file, output_file,
    batch_size, use_gpu
):
    import numpy as np
    import math
    import vcfpy
    import copy
    import time

    import tensorflow as tf
    from seqiolib import Interval, Encoder, VariantType, Variant
    from seqiolib import utils

    from pyfaidx import Fasta
    import pybedtools

    # Setup TensorFlow with optimizations
    is_gpu, strategy = setup_tensorflow(use_gpu)

    # Adjust batch size based on device
    if is_gpu and batch_size < 64:
        print(f"GPU detected: consider increasing batch size from {batch_size} to 64+ for better performance")
    elif not is_gpu and batch_size > 32:
        original_batch_size = batch_size
        batch_size = min(batch_size, 32)
        print(f"CPU mode: adjusted batch size {original_batch_size} → {batch_size}")

    def loadAndPredict(sequences, model, variants=None, batch_size=32, description="Predicting"):
        """
        Optimized prediction with batch processing and progress tracking.
        """
        start_time = time.time()
        total_sequences = len(sequences)

        # Prepare all sequences first
        X = []
        for i, sequence in enumerate(sequences):
            if variants is not None:
                sequence.replace(variants[i])
            seq_encoded = Encoder.one_hot_encode_along_channel_axis(sequence.getSequence())
            X.append(seq_encoded)

        X_array = np.array(X)

        # Batch prediction with progress tracking
        predictions = []
        last_update_time = start_time

        click.echo(f"{description} {total_sequences} sequences in batches of {batch_size}...")

        for batch_start in range(0, total_sequences, batch_size):
            batch_end = min(batch_start + batch_size, total_sequences)
            X_batch = X_array[batch_start:batch_end]

            # Predict batch
            batch_pred = model.predict(X_batch, verbose=0)
            predictions.append(batch_pred)

            # Progress update every 5 seconds
            current_time = time.time()
            if current_time - last_update_time >= 5.0:
                elapsed = current_time - start_time
                pred_per_sec = batch_end / elapsed if elapsed > 0 else 0
                click.echo(f"  {batch_end}/{total_sequences} sequences ({pred_per_sec:.1f} seq/sec)")
                last_update_time = current_time

        # Concatenate all batch predictions
        prediction = np.vstack(predictions) if len(predictions) > 1 else predictions[0]

        # Final timing
        total_time = time.time() - start_time
        avg_speed = total_sequences / total_time if total_time > 0 else 0
        click.echo(f"  Completed: {total_time:.1f}s, {avg_speed:.1f} seq/sec average")

        return prediction

    def extendIntervals(intervals, region_length, genome_file):
        left = math.ceil((region_length - 1) / 2)
        right = math.floor((region_length - 1) / 2)
        click.echo("Extending intervals left=%d, right=%d..." % (left, right))
        return list(
            map(
                pybedtoolsIntervalToInterval,
                intervals.slop(r=right, l=left, g=str(genome_file)),
            )
        )

    def getCorrectedChrom(chrom):
        if chrom.startswith("chr"):
            return chrom
        elif chrom == "MT":
            return "chrM"
        else:
            return "chr" + chrom

    def variantToPybedtoolsInterval(record):
        return pybedtools.Interval(getCorrectedChrom(record.CHROM), record.POS - 1, record.POS)

    def pybedtoolsIntervalToInterval(interval_pybed):
        return Interval(
            interval_pybed.chrom, interval_pybed.start + 1, interval_pybed.stop
        )

    # load variants
    click.echo("Loading variants...")
    records = []
    vcf_reader = vcfpy.Reader.from_path(variants_file)

    for record in vcf_reader:
        records.append(record)
    click.echo("Found %d variants" % len(records))

    if len(records) == 0:
        click.echo("No variants found. Writing file with header only and exiting...")
        vcf_writer = vcfpy.Writer.from_path(output_file, vcf_reader.header)
        vcf_writer.close()
        exit(0)
    # convert to intervals (pybedtools)
    click.echo("Convert to bed tools intervals...")
    intervals = pybedtools.BedTool(list(map(variantToPybedtoolsInterval, records)))

    # Load model with TF2 compatibility
    if strategy is not None:
        with strategy.scope():
            click.echo("Loading model within distributed strategy scope...")
            model = load_model_tf2(model_file, weights_file)
    else:
        click.echo("Loading model...")
        model = load_model_tf2(model_file, weights_file)

    input_length = model.input_shape[1]
    click.echo("Detecting interval length of %d" % input_length)
    intervals = extendIntervals(intervals, input_length, genome_file)

    # load sequence for variants
    reference = Fasta(reference_file)
    sequences_ref = []
    sequences_alt = []
    predict_avail_idx = set()

    click.echo("Load reference and try to get ref and alt.")
    alt_idx = 0
    for i in range(len(records)):
        record = records[i]
        interval = intervals[i]

        # can be problematic if we are on the edges of a chromose.
        # Workaround. It is possible to extend the intreval left or right to get the correct length
        if interval.length != input_length:
            click.echo(
                "Cannot use variant %s because of wrong size of interval %s "
                % (str(record), str(interval))
            )
            alt_idx += len(record.ALT)
            continue

        sequence_ref = utils.io.SequenceIO.readSequence(reference, interval)

        for j in range(len(record.ALT)):
            alt_record = record.ALT[j]
            variant = Variant(
                getCorrectedChrom(record.CHROM), record.POS, record.REF, alt_record.value
            )
            # INDEL
            if (
                variant.type == VariantType.DELETION
                or variant.type == VariantType.INSERTION
            ):
                # DELETION
                if variant.type == VariantType.DELETION:
                    extend = len(variant.ref) - len(variant.alt)
                    if interval.isReverse():
                        interval.position = interval.position + extend
                    else:
                        interval.position = interval.position - extend
                    interval.length = interval.length + extend
                # INSERTION
                elif variant.type == VariantType.INSERTION:
                    extend = len(variant.alt) - len(variant.ref)
                    if interval.isReverse():
                        interval.position = interval.position - extend
                    else:
                        interval.position = interval.position + extend
                    interval.length = interval.length - extend
                if interval.length > 0:
                    sequence_alt = utils.io.SequenceIO.readSequence(
                        reference, interval
                    )
                    sequence_alt.replace(variant)
                    if len(sequence_alt.sequence) == input_length:
                        # FIXME: This is a hack. it seems that for longer indels the replacement does not work
                        sequences_alt.append(sequence_alt)
                        sequences_ref.append(sequence_ref)
                        predict_avail_idx.add(alt_idx)
                    else:
                        print(
                            "Cannot use variant %s because of wrong interval %s has wrong size after InDel Correction"
                            % (str(variant), str(interval))
                        )
                else:
                    print(
                        "Cannot use variant %s because interval %s has negative size"
                        % (str(variant), str(interval))
                    )
            # SNV
            else:
                sequence_alt = copy.copy(sequence_ref)
                sequence_alt.replace(variant)
                if len(sequence_alt.sequence) == input_length:
                    # FIXME: This is a hack. it seems that for longer MNVs the replacement does not work
                    sequences_alt.append(sequence_alt)
                    sequences_ref.append(sequence_ref)
                    predict_avail_idx.add(alt_idx)
                else:
                    print(
                        "Cannot use variant %s because of wrong interval %s has wrong size after InDel Correction"
                        % (str(variant), str(interval))
                    )
            alt_idx += 1

    # Optimized batch predictions with progress tracking
    results_ref = loadAndPredict(sequences_ref, model, batch_size=batch_size, description="Predicting reference")
    results_alt = loadAndPredict(sequences_alt, model, batch_size=batch_size, description="Predicting alternative")

    num_targets = results_ref.shape[1] if len(results_alt.shape) > 1 else 1

    for task_id in range(num_targets):
        vcf_reader.header.add_info_line(
            vcfpy.OrderedDict(
                [
                    ("ID", "RegSeq%d" % task_id),
                    ("Number", "A"),
                    ("Type", "Float"),
                    (
                        "Description",
                        "Regulatory sequence prediction of the alt minus reference, output task %d"
                        % task_id,
                    ),
                ]
            )
        )

    vcf_writer = vcfpy.Writer.from_path(output_file, vcf_reader.header)

    alt_idx = 0
    predict_idx = 0
    for i in range(len(records)):
        record = records[i]
        to_add = {}
        for j in range(len(record.ALT)):
            if alt_idx in predict_avail_idx:
                for task_id in range(num_targets):
                    to_add["RegSeq%d" % task_id] = to_add.get(
                        "RegSeq%d" % task_id, []
                    ) + [round(results_alt[predict_idx][task_id] - results_ref[predict_idx][task_id], 6)]
                predict_idx += 1
            else:
                for task_id in range(num_targets):
                    to_add["RegSeq%d" % task_id] = to_add.get(
                        "RegSeq%d" % task_id, []
                    ) + [np.nan]
            alt_idx += 1

        for key, value in to_add.items():
            record.INFO[key] = value

        vcf_writer.write_record(record)
    vcf_writer.close()


if __name__ == "__main__":
    cli()
