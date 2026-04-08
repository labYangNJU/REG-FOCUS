import argparse
import pandas as pd
import os


def merge_features(chrombpnet_file, dnabert_file, output_file):
    """
    Merge ChromBPNet and DNABERT2 feature files based on variant ID.

    Parameters
    ----------
    chrombpnet_file : str
        Path to ChromBPNet feature file (TSV format).
    dnabert_file : str
        Path to DNABERT2 feature file (CSV format).
    output_file : str
        Path to save merged output file.
    """

    print("Processing files:")
    print(f"  ChromBPNet: {chrombpnet_file}")
    print(f"  DNABERT2 : {dnabert_file}")

    # Check file existence
    for f in [chrombpnet_file, dnabert_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file not found: {f}")

    try:
        # Load data
        df_chrombpnet = pd.read_csv(chrombpnet_file, sep='\t')
        df_dnabert = pd.read_csv(dnabert_file)

        # Clean and rename columns
        df_chrombpnet = df_chrombpnet.drop(
            columns=['chr', 'pos', 'allele1', 'allele2'],
            errors='ignore'
        ).rename(columns={'variant_id': 'ID'})

        # Merge on ID
        merged_df = pd.merge(df_chrombpnet, df_dnabert, on='ID', how='left')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save output
        merged_df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Merged file saved to: {output_file}")

    except Exception as e:
        print(f"[ERROR] Failed to merge files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge ChromBPNet and DNABERT2 feature files."
    )

    parser.add_argument(
        "--chrombpnet_file",
        required=True,
        help="Path to ChromBPNet feature file (TSV)"
    )

    parser.add_argument(
        "--dnabert_file",
        required=True,
        help="Path to DNABERT2 feature file (CSV)"
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to output merged CSV file"
    )

    args = parser.parse_args()

    merge_features(
        args.chrombpnet_file,
        args.dnabert_file,
        args.output_file
    )
