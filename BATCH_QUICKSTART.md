# CADD Batch Processing - Quick Start Guide

## TL;DR

```bash
# Run CADD with SLURM batch processing
./CADD_batch.sh -b -s -A YOUR_ACCOUNT -Q normal -n 1000 input.vcf.gz
```

## Common Commands

### Local Batch Processing (No SLURM)
```bash
./CADD_batch.sh -b -n 1000 input.vcf.gz
```

### SLURM Batch Processing (Recommended for Large Files)
```bash
./CADD_batch.sh -b -s -A myaccount -Q normal -j 50 -n 1000 input.vcf.gz
```

### With Custom Output and Annotations
```bash
./CADD_batch.sh -b -s -A myaccount -a -o results.tsv.gz input.vcf.gz
```

### Debug Mode (Keep Temporary Files)
```bash
./CADD_batch.sh -b -s -A myaccount -d input.vcf.gz
```

## Essential Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-b` | Enable batch mode | Required for batching |
| `-s` | Use SLURM | For cluster submission |
| `-A` | SLURM account | `-A project123` |
| `-Q` | SLURM partition | `-Q gpu` or `-Q normal` |
| `-n` | Variants per batch | `-n 1000` |
| `-j` | Max concurrent jobs | `-j 50` |
| `-a` | Include annotations | Add to output |
| `-d` | Debug mode | Keep temp files |

## Choosing Batch Size

- **SNVs only**: 1000-5000 per batch
- **Mixed variants**: 500-1000 per batch
- **Many indels**: 100-500 per batch

## Resource Requirements

### Minimum
- 8 GB RAM per job
- 2-4 CPU cores per job

### Recommended for GPU Rules (ESM, MMSplice)
- 16 GB RAM
- 4 CPU cores
- 1 GPU

## Checking Job Status

```bash
# Check running jobs
squeue -u $USER

# Check completed jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed

# Cancel all your jobs
scancel -u $USER
```

## File Locations

- **Input**: Your VCF/VCF.GZ file
- **Output**: `input_name.tsv.gz` (or specified with `-o`)
- **Logs**: `.snakemake/log/` and `slurm-*.out`
- **Temp**: `/tmp/tmp.XXXXXX/` (deleted unless `-d` used)

## Example Workflows

### Small Dataset (< 10k variants)
```bash
# No need for batch mode
./CADD.sh input.vcf.gz
```

### Medium Dataset (10k - 100k variants)
```bash
# Local batch processing
./CADD_batch.sh -b -n 1000 -c 8 input.vcf.gz
```

### Large Dataset (> 100k variants)
```bash
# SLURM batch processing
./CADD_batch.sh -b -s -A myaccount -Q normal -j 50 -n 1000 input.vcf.gz
```

### Huge Dataset (> 1M variants)
```bash
# SLURM with more jobs and larger batches
./CADD_batch.sh -b -s -A myaccount -Q normal -j 100 -n 5000 input.vcf.gz
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "SLURM account required" | Add `-A your_account` |
| Jobs failing with memory | Increase batch size or use highmem partition |
| No GPU available | Use `-Q gpu` partition |
| Too slow | Increase `-j` (max jobs) |
| Too many jobs | Decrease `-j` |

## Getting Help

```bash
# Show full help
./CADD_batch.sh -h

# Test without running (dry run)
snakemake --snakefile Snakefile_batch --dryrun output.tsv.gz
```

## Full Documentation

See `BATCH_PROCESSING.md` for complete documentation.
