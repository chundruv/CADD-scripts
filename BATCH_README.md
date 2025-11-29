# CADD Batch Processing Implementation

This implementation adds SLURM-based batch processing to CADD-scripts, enabling parallel processing of large variant datasets across cluster nodes.

## What's New

### Files Added

1. **`CADD_batch.sh`** - New wrapper script with batch processing support
2. **`Snakefile_batch`** - Modified Snakefile with batch processing rules
3. **`src/scripts/split_vcf.py`** - Script to split VCF files into batches
4. **`src/scripts/merge_batch_results.py`** - Script to merge batch results
5. **`profiles/slurm/config.yaml`** - SLURM profile configuration
6. **`profiles/slurm/slurm-status.py`** - SLURM job status checker
7. **`config/slurm_config.yaml`** - SLURM resource configuration template
8. **`BATCH_PROCESSING.md`** - Complete documentation
9. **`BATCH_QUICKSTART.md`** - Quick reference guide

### Key Features

✅ **Automatic Batch Splitting** - Splits VCF files into batches of N variants (default: 1000)
✅ **SLURM Integration** - Submit jobs to SLURM with customizable resources
✅ **Pre-scored Exclusion** - Pre-scored variants are excluded from batch processing
✅ **Parallel Processing** - Process multiple batches simultaneously
✅ **Resource Management** - Per-rule resource specifications (CPU, memory, GPU)
✅ **Automatic Merging** - Results automatically merged into single output
✅ **Backward Compatible** - Original CADD.sh workflow unchanged

## Quick Start

### 1. Basic Usage

```bash
# Local batch processing
./CADD_batch.sh -b -n 1000 input.vcf.gz

# SLURM batch processing
./CADD_batch.sh -b -s -A myaccount -Q normal input.vcf.gz
```

### 2. Configure SLURM Settings

Edit `config/slurm_config.yaml`:
```yaml
slurm_account: "your_account"
slurm_partition: "normal"
slurm_gpu_partition: "gpu"
```

### 3. Run with Custom Settings

```bash
./CADD_batch.sh \
  -b \                      # Enable batch mode
  -s \                      # Use SLURM
  -A project123 \           # Your SLURM account
  -Q normal \               # SLURM partition
  -j 50 \                   # Max 50 concurrent jobs
  -n 1000 \                 # 1000 variants per batch
  -a \                      # Include annotations
  -o results.tsv.gz \       # Output file
  input.vcf.gz              # Input file
```

## How It Works

### Workflow

1. **Prescore** - Extract pre-scored variants (excluded from batching)
2. **Split** - Split novel variants into batches
3. **Process Batches** (in parallel):
   - VEP annotation
   - ESM annotation (GPU)
   - RegSeq annotation
   - MMSplice annotation (GPU, GRCh38 only)
   - Imputation
   - Scoring
4. **Merge** - Combine batch results
5. **Join** - Add pre-scored variants
6. **Output** - Final TSV.GZ file

### Resource Allocation

| Rule | CPU | Memory | Time | GPU |
|------|-----|--------|------|-----|
| VEP | 2 | 8 GB | 3h | - |
| ESM | 4 | 16 GB | 6h | ✓ |
| RegSeq | 2 | 8 GB | 3h | - |
| MMSplice | 2 | 12 GB | 4h | ✓ |
| Score | 1 | 8 GB | 2h | - |

## Command-Line Options

### Batch Processing
- `-b` - Enable batch mode
- `-n <size>` - Variants per batch (default: 1000)

### SLURM Options
- `-s` - Use SLURM executor
- `-A <account>` - SLURM account (required)
- `-Q <partition>` - SLURM partition/queue (default: normal)
- `-j <max_jobs>` - Max concurrent jobs (default: 100)

### Standard CADD Options
- `-o <file>` - Output file
- `-g <build>` - Genome build (GRCh37/GRCh38)
- `-a` - Include annotations
- `-c <cores>` - CPU cores for Snakemake
- `-d` - Debug mode (keep temp files)

## Examples

### Example 1: Small Dataset
```bash
# < 10k variants - use original workflow
./CADD.sh input.vcf.gz
```

### Example 2: Medium Dataset
```bash
# 10k-100k variants - local batch processing
./CADD_batch.sh -b -n 1000 -c 8 input.vcf.gz
```

### Example 3: Large Dataset
```bash
# > 100k variants - SLURM batch processing
./CADD_batch.sh -b -s -A myaccount -j 50 -n 1000 input.vcf.gz
```

### Example 4: Production Run
```bash
# Full options for production
./CADD_batch.sh \
  -b -s \
  -A project123 \
  -Q gpu \
  -j 100 \
  -n 1000 \
  -g GRCh38 \
  -a \
  -o final_scores.tsv.gz \
  large_dataset.vcf.gz
```

## Performance

### Expected Speedup (100k variants)

| Mode | Time | Speedup |
|------|------|---------|
| Standard | ~24h | 1x |
| Batch (local, 8 cores) | ~12h | 2x |
| Batch (SLURM, 50 jobs) | ~3h | 8x |

Actual speedup depends on cluster resources and queue wait times.

### Batch Size Guidelines

- **SNVs**: 1000-5000 per batch
- **Mixed**: 500-1000 per batch
- **Indels**: 100-500 per batch

## Configuration Files

### SLURM Configuration
`config/slurm_config.yaml` - Customize resource allocation per rule

### SLURM Profile
`profiles/slurm/config.yaml` - Snakemake SLURM executor settings

## Monitoring Jobs

```bash
# Check running jobs
squeue -u $USER

# Check completed jobs
sacct -u $USER

# View logs
cat slurm-*.out
ls .snakemake/log/
```

## Troubleshooting

### Common Issues

1. **"SLURM account required"**
   - Solution: Add `-A your_account`

2. **Out of memory errors**
   - Solution: Increase batch size or use highmem partition

3. **GPU not available**
   - Solution: Use `-Q gpu` partition

4. **Jobs queuing too long**
   - Solution: Reduce `-j` max jobs or use different partition

### Debug Mode

```bash
# Keep temp files and show full output
./CADD_batch.sh -b -s -A account -d -p input.vcf.gz
```

## Documentation

- **`BATCH_QUICKSTART.md`** - Quick reference card
- **`BATCH_PROCESSING.md`** - Complete documentation
- **`config/slurm_config.yaml`** - Configuration template

## Architecture

### File Structure
```
CADD-scripts/
├── CADD.sh                      # Original wrapper (unchanged)
├── CADD_batch.sh                # New batch wrapper
├── Snakefile                    # Original Snakefile (unchanged)
├── Snakefile_batch              # Batch processing Snakefile
├── src/scripts/
│   ├── split_vcf.py            # VCF splitting
│   └── merge_batch_results.py  # Result merging
├── profiles/slurm/
│   ├── config.yaml             # SLURM profile
│   └── slurm-status.py         # Job status checker
└── config/
    └── slurm_config.yaml       # Resource configuration
```

### Key Design Decisions

1. **Separate Snakefile** - `Snakefile_batch` keeps original workflow intact
2. **Prescore First** - Pre-scored variants excluded from batching
3. **Conditional Batching** - Batch mode controlled by config flag
4. **Resource Specifications** - All rules have SLURM resource definitions
5. **Checkpoint-based Splitting** - Dynamic batch discovery

## Backward Compatibility

The original CADD workflow is **completely unchanged**:

```bash
# Original workflow still works exactly the same
./CADD.sh input.vcf.gz
```

Only the new `CADD_batch.sh` script enables batch processing.

## Testing

### Dry Run (Test Without Running)

```bash
snakemake output.tsv.gz \
  --snakefile Snakefile_batch \
  --configfile config/config_GRCh38_v1.7.yml \
  --config BatchMode=True BatchSize=1000 \
  --executor slurm \
  --default-resources slurm_account=myaccount slurm_partition=normal \
  --jobs 50 \
  --dryrun
```

### Test Run (Small Dataset)

```bash
# Create small test file
head -n 1000 large_file.vcf > test.vcf

# Test batch processing
./CADD_batch.sh -b -s -A account -n 100 test.vcf
```

## Next Steps

1. **Configure SLURM settings** - Edit `config/slurm_config.yaml`
2. **Test with small dataset** - Verify workflow works
3. **Optimize batch size** - Test different sizes for your data
4. **Scale up** - Process large datasets with full parallelization

## Support

- Review logs in `.snakemake/log/` and `slurm-*.out`
- Check job status with `squeue` and `sacct`
- Enable debug mode with `-d` flag
- Consult `BATCH_PROCESSING.md` for detailed troubleshooting

## Credits

CADD-scripts version 1.7 with batch processing enhancement.
Original CADD by University of Washington, Hudson-Alpha Institute, and Berlin Institute of Health.
