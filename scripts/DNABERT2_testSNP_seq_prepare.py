#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pyfaidx import Fasta
import csv
import os
from multiprocessing import Pool


def process_chunk(args):
    """
    Process a chunk of SNP lines
    
    Args:
        args (tuple): (chunk, ref_path, output_dir, base_name_with_seq, chunk_idx)
    """
    chunk, ref_path, output_dir, base_name_with_seq, chunk_idx = args
    genome = Fasta(ref_path)
    
    # Output file path: output_dir / {base_name_with_seq}_{chunk_idx}.csv
    output_file = os.path.join(output_dir, f"{base_name_with_seq}_{chunk_idx}.csv")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'REF_seq', 'ALT_seq'])

        for line in chunk:
            fields = line.strip().split('\t')
            if len(fields) != 5:  # CHROM, POS, REF, ALT, ID
                continue
            chrom, pos_str, ref_base, alt_base, _ = fields
            try:
                pos = int(pos_str)
            except ValueError:
                continue

            # Extract sequence: POS-250 to POS+250 → 501 bp
            start = pos - 251  # 0-based inclusive
            end = pos + 250    # 0-based exclusive

            try:
                ref_seq = genome[chrom][start:end].seq.upper()
            except KeyError:
                print(f"Warning: Chromosome {chrom} not found in reference genome (chunk {chunk_idx}).")
                continue

            if len(ref_seq) != 501:
                print(f"Warning: Extracted sequence for {chrom}:{pos} is {len(ref_seq)} bp, expected 501 (chunk {chunk_idx}).")
                continue

            mid_pos = 250
            if ref_seq[mid_pos] != ref_base.upper():
                print(
                    f"Warning: Middle base mismatch at {chrom}:{pos}: "
                    f"expected {ref_base}, got {ref_seq[mid_pos]} (chunk {chunk_idx})."
                )

            # Generate ALT sequence
            alt_seq = ref_seq[:mid_pos] + alt_base.upper() + ref_seq[mid_pos + 1:]
            snp_id = f"{chrom}:{pos}:{ref_base}:{alt_base}"

            writer.writerow([snp_id, ref_seq, alt_seq])


def main():
    parser = argparse.ArgumentParser(
        description="Extract 501 bp flanking sequences for SNPs from hg38. "
                    "Each input file → one output folder → chunked CSV files named {name}_seq_*.csv"
    )
    parser.add_argument('--ref', required=True,
                        help="Path to the hg38 reference genome FASTA (with .fai index)")
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help="Number of SNPs per output chunk (default: 10000)")
    parser.add_argument('--num-workers', type=int, default=10,
                        help="Number of parallel workers (default: 10)")
    parser.add_argument('input_files', nargs='+',
                        help="One or more input SNP TXT files (CHROM POS ID REF ALT)")

    args = parser.parse_args()

    # === Parameter validation ===
    num_workers = min(args.num_workers, os.cpu_count() or 1, 10)
    if num_workers < 1:
        num_workers = 1
    chunk_size = max(1, args.chunk_size)

    print(f"Using {num_workers} workers, chunk size: {chunk_size}")

    # === Process each input file ===
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found, skipping: {input_file}")
            continue

        # --- 1. Create output folder ---
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = input_basename
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing: {input_file}")
        print(f"Output folder: {os.path.abspath(output_dir)}")

        # --- 2. Read SNP lines (skip header) ---
        with open(input_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"Warning: No data in {input_file}")
            continue

        # --- 3. Split into chunks ---
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        base_name_with_seq = input_basename + "_seq"

        # --- 4. Parallel processing ---
        process_args = [
            (chunk, args.ref, output_dir, base_name_with_seq, i)
            for i, chunk in enumerate(chunks)
        ]

        print(f"Starting {len(chunks)} chunks with {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            pool.map(process_chunk, process_args)

        print(f"Completed: {input_file} → {len(chunks)} files in {output_dir}/")

    print("\nAll files processed successfully.")


if __name__ == "__main__":
    main()