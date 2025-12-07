from esm_lib import *


"""
Description: input is vep annotated vcf and a file containing all peptide sequences with Ensemble transcript IDs.
The script adds a score for frameshifts and stop gains to the info column of the vcf file. In brief, Scores for missense variants were calculated for variants annotated with the Ensembl
VEP tools' missense consequence annotation. Stop gain, stop lost, and stop retained consequences were explicitly excluded. Multiple amino acid substitutions were treated as inframe
InDel . The amino acid sequence was obtained by matching the Ensembl transcript ID with entries in a FASTA file containing amino acid sequences of all transcripts in the
Ensemble data base . Optional stop codons were removed from the sequence and the amino acid substitution was centered within a 350 amino acid window. As final score, we used the
averaged log odds ratio between the alternative
and reference amino acid employing the five different ESM-1v models "esm1v_t33_650M_UR90S_1", "esm1v_t33_650M_UR90S_2", "esm1v_t33_650M_UR90S_3", "esm1v_t33_650M_UR90S_4", and "esm1v_t33_650M_UR90S_5".
Author: thorben Maass, Max Schubach
Contact: tho.maass@uni-luebeck.de
Year:2023

OPTIMIZED VERSION: Performance improvements by loading models once, CPU optimizations, and batching efficiently
"""

import torch
from esm import pretrained
import click
import time
import os
import multiprocessing

# Check for mixed precision support
try:
    from torch.cuda.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# CPU optimization settings
def setup_cpu_optimizations():
    """Configure optimal CPU settings for PyTorch."""
    # Set number of threads for intra-op parallelism
    num_cores = multiprocessing.cpu_count()
    # Use all cores but leave some headroom
    optimal_threads = max(1, num_cores - 1)

    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(optimal_threads)

    # Enable ONEDNN/MKL optimizations if available
    if hasattr(torch.backends, 'mkldnn'):
        if hasattr(torch.backends.mkldnn, 'enabled'):
            torch.backends.mkldnn.enabled = True

    # Enable TorchScript optimizations for CPU
    torch.jit.enable_onednn_fusion(True)

    # Set environment variables for BLAS libraries
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)

    return optimal_threads


@click.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    multiple=False,
    type=click.Path(exists=True, readable=True),
    help="bgzip compressed vcf file",
)
@click.option(
    "--transcripts",
    "transcript_file",
    required=True,
    multiple=False,
    type=click.Path(exists=True, readable=True),
    help="Fasta file of peptides for all transcripts, used for esm score",
)
@click.option(
    "--model-directory",
    "model_directory",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="EMS torch model directory, usedfor torch hub",
)
@click.option(
    "--model",
    "modelsToUse",
    multiple=True,
    type=str,
    default=[
        "esm1v_t33_650M_UR90S_1",
        "esm1v_t33_650M_UR90S_2",
        "esm1v_t33_650M_UR90S_3",
        "esm1v_t33_650M_UR90S_4",
        "esm1v_t33_650M_UR90S_5",
    ],
    help="Models for download, default is all 5 models",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(writable=True),
    help="Output file, vcf file bgzip compresssed",
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=20,
    help="Batch size for esm model, default is 20",
)
def cli(input_file, transcript_file, model_directory, modelsToUse, output_file, batch_size):
    torch.hub.set_dir(model_directory)

    # Detect device and optimize accordingly
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device_type.upper()}")

    if device_type == "cpu":
        print("Configuring CPU optimizations...")
        optimal_threads = setup_cpu_optimizations()
        print(f"  Using {optimal_threads} CPU threads")
        print(f"  ONEDNN/MKL optimizations enabled")

        # Adjust batch size for CPU (smaller batches work better on CPU)
        if batch_size > 10:
            original_batch_size = batch_size
            batch_size = min(batch_size, 10)  # Cap at 10 for CPU
            print(f"  Adjusted batch size for CPU: {original_batch_size} → {batch_size}")
    else:
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # load vcf data
    vcf_data, info_pos_Feature, info_pos_ProteinPosition, info_pos_AA, info_pos_consequence = read_variants(input_file)
    # load transcripts
    transcript_info, transcript_info_ids = read_transcripts(transcript_file)
    # get missense variants
    variant_ids, transcript_ids, oAA, nAA, proteinPositions = get_missense_variants_for_esm(
        vcf_data, info_pos_Feature, info_pos_ProteinPosition, info_pos_AA, info_pos_consequence)

    print(len(variant_ids), " missense variants found")
    # create the amino acid sequence
    data, proteinPositions_mod, aa_seq = create_aa_seq(transcript_ids,
                                                       transcript_info,
                                                       transcript_info_ids,
                                                       proteinPositions,
                                                       variant_ids,
                                                       oAA,)

    # OPTIMIZATION: Load all models ONCE at the beginning
    print("Loading ESM models (this may take a while)...")
    models_and_alphabets = []
    batch_converter = None

    for k, model_name in enumerate(modelsToUse):
        print(f"  Loading model {k+1}/{len(modelsToUse)}: {model_name}")
        model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model.eval()  # disables dropout for deterministic results

        if device_type == "cuda":
            model = model.cuda()
            print(f"    Transferred to GPU")

            # OPTIMIZATION: Use mixed precision (FP16) for faster inference on Tensor Cores
            if AMP_AVAILABLE:
                print(f"    Enabled mixed precision (FP16)")

            # OPTIMIZATION: Compile model with torch.compile() if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    print(f"    Model compiled with torch.compile()")
                except Exception as e:
                    print(f"    Warning: torch.compile() failed, continuing without it: {e}")

        else:  # CPU optimizations
            print(f"    Optimizing for CPU inference...")

            # OPTIMIZATION: Use TorchScript optimization if available
            try:
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
                print(f"    Applied TorchScript optimization")
            except Exception as e:
                # If scripting fails, continue without it
                print(f"    TorchScript optimization not available: {e}")
                pass

            # Enable inference mode optimizations
            torch.set_grad_enabled(False)

        models_and_alphabets.append((model, alphabet))

        # OPTIMIZATION: Reuse the same batch_converter for all models (they share the same alphabet)
        if batch_converter is None:
            batch_converter = alphabet.get_batch_converter()

    print("All models loaded successfully!")

    # OPTIMIZATION: Process all models instead of loading them multiple times
    modelScores = []  # scores of different models

    if len(data) >= 1:
        for k, (model, alphabet) in enumerate(models_and_alphabets):
            print(f"Processing model {k+1}/{len(modelsToUse)}...")
            model_start_time = time.time()
            total_predictions = 0
            last_update_time = model_start_time

            all_scores = []  # scores for all variants of a particular model
            for t in range(0, len(data), batch_size):
                if t + batch_size >= len(data):
                    batch_data = data[t:]
                else:
                    batch_data = data[t: t + batch_size]

                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

                # OPTIMIZATION: Use inference_mode for better performance (no gradient tracking overhead)
                with torch.inference_mode():
                    if device_type == "cuda" and AMP_AVAILABLE:
                        # GPU with mixed precision
                        with autocast():
                            token_probs = torch.log_softmax(
                                model(batch_tokens.cuda())["logits"], dim=-1
                            )
                    elif device_type == "cuda":
                        # GPU without mixed precision
                        token_probs = torch.log_softmax(
                            model(batch_tokens.cuda())["logits"], dim=-1
                        )
                    else:
                        # CPU inference
                        token_probs = torch.log_softmax(
                            model(batch_tokens)["logits"], dim=-1
                        )

                # Extract scores
                for i in range(0, len(batch_data), 1):
                    if batch_data[i][1] == "NA":
                        score = 0
                        all_scores.append(float(score))
                        continue
                    else:
                        score = (
                            token_probs[
                                i,
                                proteinPositions_mod[i + t] + 1 - 1,
                                alphabet.get_idx(nAA[i + t]),
                            ]
                            - token_probs[
                                i,
                                proteinPositions_mod[i + t] + 1 - 1,
                                alphabet.get_idx(oAA[i + t]),
                            ]
                        )  # protPos+1 weil Sequenz bei 0 los geht, +1 weil esm model das so will (hebt sich eigentlich auf, aber der vollstaendigkeit halber...)
                        all_scores.append(float(score))

                # Progress tracking (print every 5 seconds)
                total_predictions += len(batch_data)
                current_time = time.time()
                if current_time - last_update_time >= 5.0:
                    elapsed = current_time - model_start_time
                    pred_per_sec = total_predictions / elapsed if elapsed > 0 else 0
                    print(f"  {total_predictions}/{len(data)} variants ({pred_per_sec:.1f} pred/sec)")
                    last_update_time = current_time

            modelScores.append(all_scores)

            # Print summary for this model
            model_elapsed = time.time() - model_start_time
            avg_pred_per_sec = total_predictions / model_elapsed if model_elapsed > 0 else 0
            print(f"  Model {k+1} complete: {model_elapsed:.1f}s, {avg_pred_per_sec:.1f} pred/sec average")

    # OPTIMIZATION: Only clear GPU cache once at the end, not in the loop
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # write out variants
    print("Writing output VCF...")
    vcf_data = variants_average_score(vcf_data, variant_ids, transcript_ids, aa_seq, len(modelsToUse), modelScores)
    write_variants(output_file, vcf_data)
    print("Done!")


if __name__ == "__main__":
    cli()
