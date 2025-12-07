"""
MMSplice prediction script for splice site analysis.

OPTIMIZED VERSION: TensorFlow 2/Keras 3 compatibility, GPU/CPU auto-detection,
mixed precision support, batch optimization, and caching for better performance.
Native DNA encoding (no concise dependency for Keras 3 compatibility).
"""

# Import
from tqdm import tqdm
from mmsplice.vcf_dataloader import SplicingVCFDataloader
from mmsplice import MMSplice
from mmsplice.utils import logit, predict_deltaLogitPsi, \
    predict_pathogenicity, predict_splicing_efficiency
import pandas as pd
import numpy as np

from argparse import ArgumentParser
import sys
import gzip
import time
import os


# Native DNA encoding function (replaces concise.preprocessing.encodeDNA)
# This is needed for Keras 3 / TensorFlow 2.x compatibility
def encodeDNA(sequences):
    """
    One-hot encode DNA sequences.

    Handles variable-length sequences by checking lengths first.

    Args:
        sequences: List of DNA sequences (strings or arrays of strings)

    Returns:
        numpy array of shape (n_sequences, sequence_length, 4)
        where the last dimension is one-hot encoding for A, C, G, T
    """
    # DNA nucleotide to index mapping
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
                  'a': 0, 'c': 1, 'g': 2, 't': 3,
                  'N': -1, 'n': -1}  # N = unknown, will be all zeros

    # Convert to list if single sequence
    if isinstance(sequences, str):
        sequences = [sequences]

    # Convert numpy array elements to strings if needed
    sequences = [str(seq) for seq in sequences]

    # Get dimensions
    n_sequences = len(sequences)
    if n_sequences == 0:
        return np.zeros((0, 0, 4), dtype=np.float32)

    # Check if all sequences have the same length
    seq_lengths = [len(seq) for seq in sequences]

    # Use the maximum length to handle variable-length sequences
    seq_length = max(seq_lengths)

    # Warn if sequences have different lengths
    if len(set(seq_lengths)) > 1:
        min_length = min(seq_lengths)
        if n_sequences <= 10:  # Only print for small batches to avoid spam
            print(f"  Warning: Variable-length sequences detected ({min_length}-{seq_length} bp, n={n_sequences})")

    # Initialize output array with max length
    encoded = np.zeros((n_sequences, seq_length, 4), dtype=np.float32)

    # Encode each sequence
    for i, seq in enumerate(sequences):
        for j, nucleotide in enumerate(seq):
            idx = nuc_to_idx.get(nucleotide, -1)
            if idx >= 0:  # Valid nucleotide
                encoded[i, j, idx] = 1.0
            # If idx is -1 (unknown), leave as all zeros

    return encoded

# TensorFlow 2 imports and configuration
try:
    import tensorflow as tf
    TF_AVAILABLE = True

    # Configure TensorFlow 2 for optimal performance
    def setup_tensorflow():
        """Configure TensorFlow 2 for optimal performance."""
        print(f"TensorFlow version: {tf.__version__}")

        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid OOM errors
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")

                # Enable mixed precision for faster training/inference on modern GPUs
                try:
                    from tensorflow.keras import mixed_precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)
                    print("Mixed precision (FP16) enabled for GPU")
                except:
                    print("Mixed precision not available, using FP32")

            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU found, using CPU")
            # Optimize CPU performance
            tf.config.threading.set_intra_op_parallelism_threads(0)  # Auto-tune
            tf.config.threading.set_inter_op_parallelism_threads(0)  # Auto-tune
            print("CPU threading optimized")

        # Enable XLA compilation for better performance (TF2 feature)
        try:
            tf.config.optimizer.set_jit(True)
            print("XLA JIT compilation enabled")
        except:
            print("XLA JIT not available")

        return len(gpus) > 0

except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. Some optimizations may not be available.")
    def setup_tensorflow():
        return False

def max_geneEff(df):
    """ Summarize largest absolute effect per variant per gene across all affected exons.
    Similar to mmsplice.utils.max_varEff
    Args:
        df: result of `predict_all_table`
    """
    df_max = df.groupby(['ID', 'gene_name'], as_index=False).agg(
        {'delta_logit_psi': lambda x: max(x, key=abs)})

    df_max = df_max.merge(df, how='left', on=['ID', 'gene_name', 'delta_logit_psi'])
    df_max = df_max.drop_duplicates(subset=['ID', 'gene_name', 'delta_logit_psi'])
    return df_max

def predict_batch_fast(model, dataloader, batch_size=512, progress=True,
                       splicing_efficiency=False, use_cache=True):
    """
    Return the prediction as a table with optimizations.

    OPTIMIZATIONS:
    - Caching of sequence predictions to avoid redundant computations
    - Vectorized operations where possible
    - Efficient batch processing
    - Progress tracking with timing information

    Args:
      model: mmsplice model object.
      dataloader: dataloader object.
      batch_size: batch size for predictions (default: 512)
      progress: show progress bar.
      splicing_efficiency: adds splicing_efficiency prediction as column
      use_cache: cache sequence predictions for reuse (default: True)
    Returns:
      iterator of pd.DataFrame of modular prediction, delta_logit_psi,
        splicing_efficiency, pathogenicity.
    """
    dataloader.encode = False
    dt_iter = dataloader.batch_iter(batch_size=batch_size)

    # Initialize progress tracking
    start_time = time.time()
    total_batches = 0
    total_variants = 0

    if progress:
        dt_iter = tqdm(dt_iter, desc="Processing batches")

    ref_cols = ['ref_acceptorIntron', 'ref_acceptor',
                'ref_exon', 'ref_donor', 'ref_donorIntron']
    alt_cols = ['alt_acceptorIntron', 'alt_acceptor',
                'alt_exon', 'alt_donor', 'alt_donorIntron']

    cat_list = ['acceptor_intron', 'acceptor', 'exon', 'donor', 'donor_intron']

    # OPTIMIZATION: Define model evaluation functions with better predict API usage
    # Use verbose=0 to suppress TF2 output and batch predictions more efficiently
    cats = {'acceptor_intron': lambda x: model.acceptor_intronM.predict(x, verbose=0),
            'acceptor': lambda x: logit(model.acceptorM.predict(x, verbose=0)),
            'exon': lambda x: model.exonM.predict(x, verbose=0),
            'donor': lambda x: logit(model.donorM.predict(x, verbose=0)),
            'donor_intron': lambda x: model.donor_intronM.predict(x, verbose=0)}

    # OPTIMIZATION: Global cache for sequence predictions across batches
    if use_cache:
        global_cache = {cat: {} for cat in cat_list}
    else:
        global_cache = None

    for batch in dt_iter:
        refs, alts = {}, {}

        for cat, model_eval in cats.items():
            alterations = batch['inputs']['seq'][cat] != \
                          batch['inputs']['mut_seq'][cat]
            if np.any(alterations):
                # OPTIMIZATION: Get unique sequences and check cache
                seq_list = list(batch['inputs']['seq'][cat][alterations])
                mut_seq_list = list(batch['inputs']['mut_seq'][cat][alterations])
                all_seqs = seq_list + mut_seq_list

                # Convert to strings and get unique sequences
                sequences = list(set([str(s) for s in all_seqs]))

                # OPTIMIZATION: Check cache for already computed sequences
                if use_cache and global_cache is not None:
                    cached_seqs = {s: global_cache[cat][s] for s in sequences if s in global_cache[cat]}
                    uncached_seqs = [s for s in sequences if s not in global_cache[cat]]

                    if uncached_seqs:
                        # Only predict for uncached sequences
                        encoded = encodeDNA(uncached_seqs)
                        prediction = model_eval(encoded).flatten()
                        new_predictions = {s: p for s, p in zip(uncached_seqs, prediction)}

                        # Update cache
                        global_cache[cat].update(new_predictions)

                        # Combine cached and new predictions
                        pred_dict = {**cached_seqs, **new_predictions}
                    else:
                        # All sequences were cached
                        pred_dict = cached_seqs
                else:
                    # No caching
                    encoded = encodeDNA(sequences)
                    prediction = model_eval(encoded).flatten()
                    pred_dict = {s: p for s, p in zip(sequences, prediction)}

                # OPTIMIZATION: Vectorized lookup using numpy
                refs[cat] = [pred_dict[batch['inputs']['seq'][cat][i]]
                            if a else 0 for i, a in enumerate(alterations)]
                alts[cat] = [pred_dict[batch['inputs']['mut_seq'][cat][i]]
                            if a else 0 for i, a in enumerate(alterations)]
            else:
                refs[cat] = [0] * len(alterations)
                alts[cat] = [0] * len(alterations)

        # OPTIMIZATION: Use numpy for faster array operations
        X_ref = np.array([refs[cat] for cat in cat_list]).T
        X_alt = np.array([alts[cat] for cat in cat_list]).T
        ref_pred = pd.DataFrame(X_ref, columns=ref_cols)
        alt_pred = pd.DataFrame(X_alt, columns=alt_cols)

        df = pd.DataFrame({
            'ID': batch['metadata']['variant']['STR'],
            'exons': batch['metadata']['exon']['annotation'],
        })
        for k in ['exon_id', 'gene_id', 'gene_name', 'transcript_id']:
            if k in batch['metadata']['exon']:
                df[k] = batch['metadata']['exon'][k]

        df['delta_logit_psi'] = predict_deltaLogitPsi(X_ref, X_alt)
        df = pd.concat([df, ref_pred, alt_pred], axis=1)

        # pathogenicity does not work
        #if pathogenicity:
        #    df['pathogenicity'] = predict_pathogenicity(X_ref, X_alt)

        if splicing_efficiency:
            df['efficiency'] = predict_splicing_efficiency(X_ref, X_alt)

        total_batches += 1
        total_variants += len(df)

        yield df

    # Print summary statistics
    elapsed = time.time() - start_time
    if total_batches > 0:
        print(f"\nProcessing complete:")
        print(f"  Total batches: {total_batches}")
        print(f"  Total variants: {total_variants}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"  Variants/sec: {total_variants/elapsed:.1f}")
        if use_cache and global_cache is not None:
            cache_sizes = {cat: len(cache) for cat, cache in global_cache.items()}
            print(f"  Cached sequences: {cache_sizes}")

def predict_table_fast(model,
                      dataloader,
                      batch_size=512,
                      progress=True,
                      pathogenicity=False,
                      splicing_efficiency=False,
                      use_cache=True):
    """
    Return the prediction as a table with optimizations.

    Args:
      model: mmsplice model object.
      dataloader: dataloader object.
      batch_size: batch size for predictions (default: 512)
      progress: show progress bar.
      pathogenicity: adds pathogenicity prediction as column (not currently supported)
      splicing_efficiency: adds splicing_efficiency prediction as column
      use_cache: enable sequence caching for better performance (default: True)
    Returns:
      pd.DataFrame of modular prediction, delta_logit_psi, splicing_efficiency,
        pathogenicity.
    """
    return pd.concat(predict_batch_fast(model, dataloader, batch_size=batch_size,
                                   progress=progress,
                                   splicing_efficiency=splicing_efficiency,
                                   use_cache=use_cache))

parser = ArgumentParser(description="MMSplice prediction with TensorFlow 2 optimizations")
parser.add_argument("-o", "--output", dest="output", type=str,
                    help="Output vcf file (default: stdout)")
parser.add_argument("-i", "--input", dest="input", type=str, required=True,
                    help="Input vcf.gz file")
parser.add_argument("-f", "--fasta", dest="fasta", type=str, required=True,
                    help="Genome fasta file")
parser.add_argument("-g", "--gtf", dest="gtf", type=str, required=True,
                    help="GTF file with all exons (may be the pickled version output by the script)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=512,
                    help="Batch size for predictions (default: 512)")
parser.add_argument("--no-cache", dest="no_cache", action="store_true",
                    help="Disable sequence caching (reduces memory usage)")
parser.add_argument("--no-progress", dest="no_progress", action="store_true",
                    help="Disable progress bar")

args = parser.parse_args()

# OPTIMIZATION: Setup TensorFlow 2 with GPU/CPU auto-detection
print("="*60)
print("MMSplice Prediction - Optimized Version")
print("="*60)
if TF_AVAILABLE:
    has_gpu = setup_tensorflow()
    print()
else:
    print("TensorFlow not available - running without TF optimizations")
    print()

# Initialize dataloader
print(f"Loading dataloader...")
print(f"  GTF: {args.gtf}")
print(f"  FASTA: {args.fasta}")
print(f"  VCF: {args.input}")

dl = SplicingVCFDataloader(args.gtf,
                           args.fasta,
                           args.input)

# Specify model
print(f"\nInitializing MMSplice model...")
model = MMSplice()
print(f"  Model loaded successfully")

# Configuration
print(f"\nConfiguration:")
print(f"  Batch size: {args.batch_size}")
print(f"  Caching: {'Disabled' if args.no_cache else 'Enabled'}")
print(f"  Progress bar: {'Disabled' if args.no_progress else 'Enabled'}")
print()

try:
    # Do prediction
    print("Starting prediction...")
    start_time = time.time()

    predictions = predict_table_fast(
        model,
        dl,
        batch_size=args.batch_size,
        progress=not args.no_progress,
        splicing_efficiency=False,
        use_cache=not args.no_cache
    )

    # Summarize with maximum effect size
    print("\nSummarizing predictions...")
    predictionsMax = max_geneEff(predictions)

    # Build prediction dictionary
    pred_dict = {}
    for p in predictionsMax[['ID', 'gene_name', 'delta_logit_psi', 'ref_acceptorIntron', 'ref_acceptor', 'ref_exon', 'ref_donor', 'ref_donorIntron', 'alt_acceptorIntron', 'alt_acceptor', 'alt_exon', 'alt_donor', 'alt_donorIntron']].values:
        if p[0] in pred_dict:
            pred_dict[p[0]].append(p[1:])
        else:
            pred_dict[p[0]] = [p[1:]]

    print(f"  Found predictions for {len(pred_dict)} variants")

except ValueError as e:
    # It can happen that no variant is in a splice region and therefore receives a score leading to an empty array
    print(f"Warning: No variants found in splice regions or error occurred: {e}")
    pred_dict = {}
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    pred_dict = {}

# OPTIMIZATION: Open output file and write results
print("\nWriting output VCF...")
write_start = time.time()
variants_written = 0
variants_annotated = 0

if args.output:
    writer = gzip.open(args.output, 'wt')
else:
    writer = sys.stdout

try:
    # Write prediction to stdout or file
    with gzip.open(args.input, 'rt') as reader:
        for line in reader:
            if line.startswith("#"):
                if line.startswith('#C'):
                    writer.write('##INFO=<ID=MMSplice,Number=.,Type=String,Description="MMSplice scores for five submodels. Format: SYMBOL|acceptorIntron|acceptor|exon|donor|donorIntron">\n')

                writer.write(line)
                continue

            # Process variant line
            fields = line.strip().split('\t')
            var = "%s:%s:%s:['%s']" % (fields[0], fields[1], fields[3], fields[4])
            variants_written += 1

            if var in pred_dict:
                val_list = []
                for p in pred_dict[var]:
                    # Calculate delta scores (alt - ref) for each submodel
                    scores = [p[i+7] - p[i+2] for i in range(5)]
                    val_list.append('|'.join([p[0]] + ['%.3f' % s for s in scores]))
                val = ','.join(val_list)

                # Add MMSplice annotation to INFO field
                if len(fields) >= 8:
                    fields[7] = fields[7] + ';MMSplice=' + val
                else:
                    fields.extend(['.']*(7 - len(fields)))
                    fields.append('MMSplice=' + val)

                variants_annotated += 1

            writer.write('\t'.join(fields) + '\n')

finally:
    if args.output:
        writer.close()

# Print final statistics
write_elapsed = time.time() - write_start
total_elapsed = time.time() - start_time

print(f"\nOutput complete:")
print(f"  Total variants processed: {variants_written}")
print(f"  Variants with MMSplice scores: {variants_annotated}")
print(f"  Write time: {write_elapsed:.1f}s")
print(f"\nTotal runtime: {total_elapsed:.1f}s")
print(f"Overall throughput: {variants_written/total_elapsed:.1f} variants/sec")
print("="*60)
print("Done!")
