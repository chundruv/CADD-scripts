"""
Description: input is vep annotated vcf and a file containing all peptide sequences with Ensemble transcript IDs.
The script adds a score for frameshifts and stop gains to the info column of the vcf file. In brief, Scores for frameshift or stop gain variants were calculated
for variants annotated with the Ensembl VEP tools' frameshift
or stop gain consequence annotation. Amino acid sequences of reference alleles were obtained as described above, with the only difference that a window of 250 amino acids was used.
Log transformed probabilities were calculated and summed up for the entire reference amino acid sequence.
Calculation of a score for the alternative allele was carried out based on the entire reference amino acid sequence. Here, we summed up log probabilities of the reference
sequences amino acids up to the point where the frameshift or stop gain occurred as obtained from the Ensembl VEP tools' annotations. For every amino acid that is lost due
to the frameshift or stop gain, we used the median of log transformed probabilities calculated from all possible amino acids at each individual position in the remaining sequence
and added them to the sum corresponding to the alternative allele.
The average of logs odds ratios between the reference and alternative sequences from the five models  was than used as a final score.

Author: Thorben Maass, Max Schubach
Contact: tho.maass@uni-luebeck.de
Year:2023

OPTIMIZED VERSION: Performance improvements by loading models once, building transcript lookup dict, and batching efficiently
"""

import warnings
import numpy as np
from Bio.bgzf import BgzfReader, BgzfWriter
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
def cli(
    input_file, transcript_file, model_directory, modelsToUse, output_file, batch_size
):
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

    # get information from vcf file with SNVs and write them into lists (erstmal Bsp, später automatisch aus info zeile extrahieren)
    vcf_file_data = BgzfReader(input_file, "r")  # TM_example.vcf.gz
    vcf_data = []
    for line in vcf_file_data:
        vcf_data.append(line)

    info_pos_Feature = False  # TranscriptID
    info_pos_ProteinPosition = False  # resdidue in protein that is mutated
    info_pos_AA = False  # mutation from aa (amino acid) x to y
    info_pos_consequence = False
    # identify positions of annotations importnat for esm score
    for line in vcf_data:
        if line[0:7] == "##INFO=":
            info = line.split("|")
            for i in range(0, len(info), 1):
                if info[i] == "Feature":
                    info_pos_Feature = i
                if info[i] == "Protein_position":
                    info_pos_ProteinPosition = i
                if info[i] == "Amino_acids":
                    info_pos_AA = i
                if info[i] == "Consequence":
                    info_pos_consequence = i
            break

    # extract annotations important for esm score, "NA" for non-coding variants
    variant_ids = []
    transcript_id = []
    oAA = []
    nAA = []
    protPosStart = []
    protPosEnd = []
    protPos_mod = []
    cons = []

    for variant in vcf_data:
        if variant[0:1] != "#":
            variant_entry = variant.split(",")
            for i in range(0, len(variant_entry), 1):
                variant_info = variant_entry[i].split("|")
                consequences = variant_info[info_pos_consequence].split("&")
                if (
                    "frameshift_variant" in consequences
                    or "stop_gained" in consequences
                ) and len(variant_info[info_pos_AA].split("/")) == 2:
                    variant_ids.append(variant_entry[0].split("|")[0])
                    transcript_id.append("transcript:" + variant_info[info_pos_Feature])
                    cons.append(variant_info[info_pos_consequence].split("&"))
                    oAA.append(
                        variant_info[info_pos_AA].split("/")[0]
                    )  # can also be "-" if there is an insertion
                    nAA.append(variant_info[info_pos_AA].split("/")[1])
                    if (
                        "-" in variant_info[info_pos_ProteinPosition].split("/")[0]
                    ):  # in case of frameshifts, vep only gives X as the new aa
                        protPosStart.append(
                            int(
                                variant_info[info_pos_ProteinPosition]
                                .split("/")[0]
                                .split("-")[0]
                            )
                        )
                        protPosEnd.append(
                            int(
                                variant_info[info_pos_ProteinPosition]
                                .split("/")[0]
                                .split("-")[1]
                            )
                        )
                    else:
                        protPosStart.append(
                            int(variant_info[info_pos_ProteinPosition].split("/")[0])
                        )
                        protPosEnd.append(
                            int(variant_info[info_pos_ProteinPosition].split("/")[0])
                        )
                    protPos_mod.append(False)

    # OPTIMIZATION: Build transcript lookup dictionary for O(1) access
    print("Building transcript lookup dictionary...")
    transcript_data = open(
        transcript_file, "r"
    )  # <Pfad zu "Homo_sapiens.GRCh38.pep.all.fa" >
    transcript_info_entries = transcript_data.read().split(
        ">"
    )  # evtl erstes > in file weglöschen
    transcript_data.close()

    # Build dictionary mapping transcript ID to (full_info, sequence)
    transcript_dict = {}
    for entry in transcript_info_entries:
        if entry != "":
            entry_parts = entry.split(" ")
            if len(entry_parts) >= 5:
                transcript_id_raw = entry_parts[4]
                # Remove version of ENST ID for comparison with vep annotation
                point_pos = transcript_id_raw.find(".")
                if point_pos != -1:
                    transcript_id_clean = transcript_id_raw[:point_pos]
                else:
                    transcript_id_clean = transcript_id_raw

                # Store the sequence (last part of entry)
                transcript_dict[transcript_id_clean] = entry_parts

    # create list with aa_seq_refs of transcript_ids
    aa_seq_ref = []
    totalNumberOfStopCodons = []
    numberOfStopCodons = []
    numberOfStopCodonsInIndel = []

    # OPTIMIZATION: Use dictionary lookup instead of nested loop
    for j in range(0, len(transcript_id), 1):
        transcript_key = transcript_id[j]  # e.g., "transcript:ENST00012345"

        # Look up in dictionary (O(1) instead of O(n))
        if transcript_key in transcript_dict:
            temp_seq = transcript_dict[transcript_key][-1]

            # prepare Seq remove remainings of header
            for k in range(0, len(temp_seq), 1):
                if temp_seq[k] != "\n":
                    k = k + 1
                else:
                    k = k + 1
                    temp_seq = temp_seq[k:]
                    break

            # prepare seq (remove /n)
            forbidden_chars = "\n"
            for char in forbidden_chars:
                temp_seq = temp_seq.replace(char, "")

            # count stop codons in seq before site of mutation
            numberOfStopCodons.append(0)
            if "*" in temp_seq:
                for k in range(0, len(temp_seq), 1):
                    if temp_seq[k] == "*" and k < protPosStart[j]:
                        numberOfStopCodons[j] = numberOfStopCodons[j] + 1

            # count stop codons in Indel
            numberOfStopCodonsInIndel.append(0)
            if "*" in temp_seq:
                for k in range(0, len(temp_seq), 1):
                    if (
                        temp_seq[k] == "*"
                        and k >= protPosStart[j]
                        and k < protPosEnd[j]
                    ):
                        numberOfStopCodonsInIndel[j] = (
                            numberOfStopCodonsInIndel[j] + 1
                        )

            # count stop codons in seq
            totalNumberOfStopCodons.append(0)
            if "*" in temp_seq:
                for k in range(0, len(temp_seq), 1):
                    if temp_seq[k] == "*":
                        totalNumberOfStopCodons[j] = totalNumberOfStopCodons[j] + 1

            # remove additional stop codons (remove *)
            forbidden_chars = "*"
            for char in forbidden_chars:
                temp_seq = temp_seq.replace(char, "")

            aa_seq_ref.append(temp_seq)
        else:
            # Transcript not found
            aa_seq_ref.append("NA")
            numberOfStopCodons.append(9999)
            totalNumberOfStopCodons.append(9999)
            numberOfStopCodonsInIndel.append(9999)

    conseq = []
    aa_seq_alt = []
    for j in range(0, len(aa_seq_ref), 1):
        if aa_seq_ref[j] == "NA":
            aa_seq_alt.append("NA")
            conseq.append("NA")
        elif "*" in nAA[j] or "X" in nAA[j]:  # stop codon gained or complete frameshift
            aa_seq_alt.append(aa_seq_ref[j])  # add alt seq without stop codon
            conseq.append("FS")
        else:
            aa_seq_alt.append("NA")
            conseq.append("NA")
            warnings.warn(
                "there is a problem with the ensembl data base and vep. The ESMframesift score of this variant will be artificially set to 0. Affected transcript is "
                + str(transcript_id[j])
            )

    # prepare data array for esm model
    window = 250
    data_ref = []
    for i in range(0, len(transcript_id), 1):
        if len(aa_seq_ref[i]) < window:
            data_ref.append((transcript_id[i], aa_seq_ref[i]))
            protPos_mod[i] = protPosStart[i] - numberOfStopCodons[i]

        elif (
            (len(aa_seq_ref[i]) >= window)
            and (
                protPosStart[i] - numberOfStopCodons[i] + 1 + window / 2
                <= len(aa_seq_ref[i])
            )
            and (protPosStart[i] - numberOfStopCodons[i] + 1 - window / 2 >= 1)
        ):
            data_ref.append(
                (
                    transcript_id[i],
                    aa_seq_ref[i][
                        protPosStart[i]
                        - numberOfStopCodons[i]
                        - int(window / 2) : protPosStart[i]
                        - numberOfStopCodons[i]
                        + int(window / 2)
                    ],
                )
            )  # esm model can only handle 1024 amino acids, so if the sequence is longer , just the sequece around the mutaion i
            protPos_mod[i] = int(
                len(
                    aa_seq_ref[i][
                        protPosStart[i]
                        - numberOfStopCodons[i]
                        - int(window / 2) : protPosStart[i]
                        - numberOfStopCodons[i]
                        + int(window / 2)
                    ]
                )
                / 2
            )

        elif (
            len(aa_seq_ref[i]) >= window
            and protPosStart[i] - numberOfStopCodons[i] + 1 - window / 2 < 1
        ):
            data_ref.append((transcript_id[i], aa_seq_ref[i][:window]))
            protPos_mod[i] = protPosStart[i] - numberOfStopCodons[i]

        else:
            data_ref.append((transcript_id[i], aa_seq_ref[i][-window:]))
            protPos_mod[i] = (
                protPosStart[i] - numberOfStopCodons[i] - (len(aa_seq_ref[i]) - window)
            )

    data_alt = []

    for i in range(0, len(transcript_id), 1):
        if len(aa_seq_alt[i]) < window:
            data_alt.append((transcript_id[i], aa_seq_alt[i]))

        elif (
            (len(aa_seq_alt[i]) >= window)
            and (
                protPosStart[i] - numberOfStopCodons[i] + 1 + window / 2
                <= len(aa_seq_alt[i])
            )
            and (protPosStart[i] - numberOfStopCodons[i] + 1 - window / 2 >= 1)
        ):
            data_alt.append(
                (
                    transcript_id[i],
                    aa_seq_alt[i][
                        protPosStart[i]
                        - numberOfStopCodons[i]
                        - int(window / 2) : protPosStart[i]
                        - numberOfStopCodons[i]
                        + int(window / 2)
                    ],
                )
            )  # esm model can only handle 1024 amino acids, so if the sequence is longer , just the sequece around the mutaion i

        elif (
            len(aa_seq_alt[i]) >= window
            and protPosStart[i] - numberOfStopCodons[i] + 1 - window / 2 < 1
        ):
            data_alt.append((transcript_id[i], aa_seq_alt[i][:window]))

        else:
            data_alt.append((transcript_id[i], aa_seq_alt[i][-window:]))

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

            # OPTIMIZATION: Use channels_last memory format for better CPU performance
            try:
                # Note: ESM models may not support channels_last, so we wrap in try-except
                if hasattr(model, 'to'):
                    # Apply ONEDNN graph optimization
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

    # OPTIMIZATION: Process all models for ref/alt instead of loading models multiple times
    ref_scores_all_models = []
    alt_scores_all_models = []

    if len(data_ref) >= 1:
        for k, (model, alphabet) in enumerate(models_and_alphabets):
            print(f"Processing model {k+1}/{len(modelsToUse)}...")
            model_start_time = time.time()
            total_predictions = 0
            last_update_time = model_start_time

            # Process reference sequences
            seq_scores_ref = []
            for t in range(0, len(data_ref), batch_size):
                if t + batch_size > len(data_ref):
                    batch_data = data_ref[t:]
                else:
                    batch_data = data_ref[t : t + batch_size]

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

                # Extract scores for ref sequences
                for i in range(0, len(batch_data), 1):
                    if conseq[i + t] == "FS":
                        score = 0
                        for y in range(0, len(batch_data[i][1]), 1):
                            score = (
                                score
                                + token_probs[
                                    i,
                                    y + 1,
                                    alphabet.get_idx(batch_data[i][1][y]),
                                ]
                            )
                        seq_scores_ref.append(float(score))
                    elif conseq[i + t] == "NA":
                        score = 999  # sollte nacher rausgeschissen werden, kein score sollte -999 sein
                        seq_scores_ref.append(float(score))

                # Progress tracking (print every 5 seconds)
                total_predictions += len(batch_data)
                current_time = time.time()
                if current_time - last_update_time >= 5.0:
                    elapsed = current_time - model_start_time
                    pred_per_sec = total_predictions / elapsed if elapsed > 0 else 0
                    print(f"  REF: {total_predictions}/{len(data_ref)} variants ({pred_per_sec:.1f} pred/sec)")
                    last_update_time = current_time

            ref_scores_all_models.append(seq_scores_ref)

            # Reset progress tracking for alt sequences
            total_predictions = 0
            last_update_time = time.time()

            # Process alternative sequences
            seq_scores_alt = []
            for t in range(0, len(data_alt), batch_size):
                if t + batch_size > len(data_alt):
                    batch_data = data_alt[t:]
                else:
                    batch_data = data_alt[t : t + batch_size]

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

                # Extract scores for alt sequences
                for i in range(0, len(batch_data), 1):
                    if conseq[i + t] == "FS":
                        score = 0
                        for y in range(0, len(batch_data[i][1]), 1):
                            if y < protPos_mod[i + t]:
                                score = (
                                    score
                                    + token_probs[
                                        i,
                                        y + 1,
                                        alphabet.get_idx(batch_data[i][1][y]),
                                    ]
                                )
                            else:
                                # OPTIMIZATION: Vectorize median calculation
                                # Collect all amino acid scores at once
                                aa_scores_tensor = torch.cat([
                                    token_probs[i, y + 1, 4:24],  # Standard amino acids
                                    token_probs[i, y + 1, 26:27]  # Selenocysteine
                                ])
                                median = torch.median(aa_scores_tensor)
                                score = score + median

                        seq_scores_alt.append(float(score))
                    elif conseq[i + t] == "NA":
                        score = 0
                        seq_scores_alt.append(float(score))

                # Progress tracking (print every 5 seconds)
                total_predictions += len(batch_data)
                current_time = time.time()
                if current_time - last_update_time >= 5.0:
                    elapsed = current_time - model_start_time
                    pred_per_sec = total_predictions / elapsed if elapsed > 0 else 0
                    print(f"  ALT: {total_predictions}/{len(data_alt)} variants ({pred_per_sec:.1f} pred/sec)")
                    last_update_time = current_time

            alt_scores_all_models.append(seq_scores_alt)

            # Print summary for this model
            model_elapsed = time.time() - model_start_time
            total_model_predictions = len(data_ref) + len(data_alt)
            avg_pred_per_sec = total_model_predictions / model_elapsed if model_elapsed > 0 else 0
            print(f"  Model {k+1} complete: {model_elapsed:.1f}s, {avg_pred_per_sec:.1f} pred/sec average")

    # OPTIMIZATION: Only clear GPU cache once at the end, not in the loop
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # Convert to numpy arrays for easier manipulation
    ref_alt_scores = [np.array(ref_scores_all_models), np.array(alt_scores_all_models)]
    np_array_scores = np.array(ref_alt_scores)
    np_array_score_diff = np_array_scores[1] - np_array_scores[0]

    # write scores in cvf. file

    # identify positions of annotations important for esm score
    header_end = False
    for i in range(0, len(vcf_data), 1):
        if vcf_data[i][0:6] == "#CHROM":
            vcf_data[i - 1] = (
                vcf_data[i - 1]
                + "##INFO=<ID=EsmScoreFrameshift"
                + ',Number=.,Type=String,Description="esmScore for one submodels. Format: esmScore">\n'
            )
            header_end = i
            break

    for i in range(header_end + 1, len(vcf_data), 1):
        j = 0
        while j < len(variant_ids):
            if vcf_data[i].split("|")[0] == variant_ids[j]:
                # count number of vep entires per variant that result in an esm score (i.e. with consequence "missense")
                numberOfEsmScoresPerVariant = 0
                for l in range(j, len(variant_ids), 1):
                    if vcf_data[i].split("|")[0] == variant_ids[l]:
                        numberOfEsmScoresPerVariant = numberOfEsmScoresPerVariant + 1
                    else:
                        break

                # annotate vcf line with esm scores
                vcf_data[i] = (
                    vcf_data[i][:-1] + ";EsmScoreFrameshift" + "=" + vcf_data[i][-1:]
                )
                for h in range(0, numberOfEsmScoresPerVariant, 1):
                    if aa_seq_ref[j + h] != "NA":
                        average_score = 0
                        for k in range(0, len(modelsToUse), 1):
                            average_score = average_score + float(
                                np_array_score_diff[k][j + h]
                            )
                        average_score = average_score / len(modelsToUse)
                        vcf_data[i] = (
                            vcf_data[i][:-1]
                            + str(transcript_id[j + h][11:])
                            + "|"
                            + str(round(float(average_score), 3))
                            + vcf_data[i][-1:]
                        )
                    else:
                        vcf_data[i] = (
                            vcf_data[i][:-1]
                            + str(transcript_id[j + h][11:])
                            + "|"
                            + "NA"
                            + vcf_data[i][-1:]
                        )

                    if h != numberOfEsmScoresPerVariant - 1:
                        vcf_data[i] = vcf_data[i][:-1] + "," + vcf_data[i][-1:]

                j = j + numberOfEsmScoresPerVariant
            else:
                j = j + 1

    print("Writing output VCF...")
    vcf_file_output = BgzfWriter(output_file, "w")
    for line in vcf_data:
        vcf_file_output.write(line)

    vcf_file_output.close()
    print("Done!")


if __name__ == "__main__":
    cli()
