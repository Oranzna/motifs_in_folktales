import os
from argparse import ArgumentParser

import pandas as pd


def parse_arguments():
    parser = ArgumentParser(
        description="Convert yes/no motif answers in an Excel file to binary 1/0 values."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the input Excel file with yes/no motif answers.",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Path where the processed Excel file should be saved.",
    )

    parser.add_argument(
        "--start_col",
        type=int,
        default=1,
        help="First motif column index to process. Default: 1, meaning the second column.",
    )

    parser.add_argument(
        "--end_col",
        type=int,
        default=16,
        help="Column index where processing stops. Default: 16, meaning columns 1 through 15 are processed.",
    )

    return parser.parse_args()


def process_excel(input_file, output_file, start_col=1, end_col=16):
    df = pd.read_excel(input_file)

    for col in df.columns[start_col:end_col]:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].apply(
            lambda x: 1 if x.startswith("yes") else (0 if x.startswith("no") else "")
        )

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_excel(output_file, index=False)

    print(f"Processed Excel file saved as '{output_file}'")


if __name__ == "__main__":
    args = parse_arguments()

    process_excel(
        input_file=args.input_file,
        output_file=args.output_file,
        start_col=args.start_col,
        end_col=args.end_col,
    )