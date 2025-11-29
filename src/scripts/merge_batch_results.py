#!/usr/bin/env python

"""
Merge batch result files into a single output file.
Handles TSV.GZ format, preserves header from first file.
"""

import argparse
import gzip
import sys


def merge_results(batch_files, output_file):
    """
    Merge batch result files.

    Args:
        batch_files: List of batch result files to merge
        output_file: Output file path

    Returns:
        Number of variants merged
    """
    variant_count = 0
    header_written = False

    with gzip.open(output_file, 'wt') as outfile:
        for batch_file in batch_files:
            with gzip.open(batch_file, 'rt') as infile:
                for line in infile:
                    # Handle header
                    if line.startswith('#'):
                        if not header_written:
                            outfile.write(line)
                        # Skip headers from subsequent files
                        continue

                    # Write variant line
                    outfile.write(line)
                    variant_count += 1

                header_written = True

    return variant_count


def main():
    parser = argparse.ArgumentParser(
        description='Merge batch result files into single output'
    )
    parser.add_argument(
        'batch_list',
        help='File containing list of batch result files (one per line)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output merged file (will be gzipped)'
    )

    args = parser.parse_args()

    # Read batch file list
    with open(args.batch_list) as f:
        batch_files = [line.strip() for line in f if line.strip()]

    print(f"Merging {len(batch_files)} batch files...", file=sys.stderr)

    variant_count = merge_results(batch_files, args.output)

    print(f"Merged {variant_count} variants to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
