#!/bin/bash
# SLURM submission script for Snakemake

# Parse Snakemake arguments
jobscript=$1

# Set default resources (can be overridden by Snakemake)
ACCOUNT=${SLURM_ACCOUNT:-""}
PARTITION=${SLURM_PARTITION:-"normal"}
CPUS=${SNAKEMAKE_THREADS:-1}
MEM=${SNAKEMAKE_RESOURCES_MEM_MB:-8000}
TIME=${SNAKEMAKE_RESOURCES_RUNTIME:-120}
EXTRA=${SNAKEMAKE_RESOURCES_SLURM_EXTRA:-""}

# Build sbatch command
sbatch_cmd="sbatch"

# Add account if specified
if [ -n "$ACCOUNT" ]; then
    sbatch_cmd="$sbatch_cmd -A $ACCOUNT"
fi

# Add partition
sbatch_cmd="$sbatch_cmd -p $PARTITION"

# Add resources
sbatch_cmd="$sbatch_cmd -c $CPUS --mem=${MEM}M -t $TIME"

# Add job name from Snakemake rule
if [ -n "$SNAKEMAKE_RULE" ]; then
    sbatch_cmd="$sbatch_cmd -J $SNAKEMAKE_RULE"
fi

# Add extra SLURM parameters (e.g., GPU)
if [ -n "$EXTRA" ]; then
    sbatch_cmd="$sbatch_cmd $EXTRA"
fi

# Submit the job script
$sbatch_cmd $jobscript
