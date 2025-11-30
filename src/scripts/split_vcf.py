#!/usr/bin/env python

"""
Split VCF file into batches of N variants.
Preserves header in each batch file.
"""

import argparse
import gzip
import os
import sys


def open_file(filename, mode='r'):
    """Open regular or gzipped file."""
    if filename.endswith('.gz'):
        if 'r' in mode:
            return gzip.open(filename, 'rt')
        else:
            return gzip.open(filename, 'wt')
    else:
        return open(filename, mode)


def split_vcf(input_file, output_dir, batch_size, output_prefix):
    """
    Split VCF file into batches.

    Args:
        input_file: Path to input VCF file (can be .gz)
        output_dir: Directory to write batch files
        batch_size: Number of variants per batch
        output_prefix: Prefix for output files

    Returns:
        List of batch file paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_files = []
    header_lines = []
    variant_count = 0
    batch_num = 0
    current_batch = None

    with open_file(input_file) as infile:
        for line in infile:
            # Collect header lines
            if line.startswith('#'):
                header_lines.append(line)
            else:
                # Start new batch if needed
                if variant_count % batch_size == 0:
                    if current_batch:
                        current_batch.close()

                    batch_num += 1
                    batch_file = os.path.join(output_dir, "{}.batch_{:04d}.vcf".format(output_prefix, batch_num))
                    batch_files.append(batch_file)
                    current_batch = open(batch_file, 'w')

                    # Write header to new batch
                    for header_line in header_lines:
                        current_batch.write(header_line)

                # Write variant line
                current_batch.write(line)
                variant_count += 1

    if current_batch:
        current_batch.close()

    # Write batch list file
    batch_list_file = os.path.join(output_dir, "{}.batch_list.txt".format(output_prefix))
    with open(batch_list_file, 'w') as f:
        for batch_file in batch_files:
            f.write("{}\n".format(batch_file))

    return batch_files, batch_list_file


def main():
    parser = argparse.ArgumentParser(
        description='Split VCF file into batches of N variants'
    )
    parser.add_argument(
        'input',
        help='Input VCF file (can be .vcf or .vcf.gz)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        help='Output directory for batch files'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=1000,
        help='Number of variants per batch (default: 1000)'
    )
    parser.add_argument(
        '-p', '--prefix',
        default='batch',
        help='Prefix for output batch files (default: batch)'
    )

    args = parser.parse_args()

    sys.stderr.write("Splitting {} into batches of {} variants...\n".format(
        args.input, args.batch_size))

    batch_files, batch_list = split_vcf(
        args.input,
        args.output_dir,
        args.batch_size,
        args.prefix
    )

    sys.stderr.write("Created {} batch files\n".format(len(batch_files)))
    sys.stderr.write("Batch list written to: {}\n".format(batch_list))

    # Output batch list file path for Snakemake
    print(batch_list)


if __name__ == '__main__':
    main()
