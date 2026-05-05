# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Remap and copy map files based on a CSV manifest.

For each row in the CSV, the script reads the file with name `original_filename`
from the source path and copies it to `<target_path>/<csv_stem>/<new_filename>`.
The CSV itself is also copied to the same target subdirectory under the name
`manifest.csv`.
"""

import argparse
import csv
import shutil
from pathlib import Path


def remap_files(source_path: Path, csv_path: Path, target_path: Path) -> None:
    # Output directory: target_path / <csv stem>
    out_dir = target_path / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"new_filename", "original_filename"}
        missing_cols = required_cols - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(
                f"CSV is missing required columns: {missing_cols}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            original = source_path / row["original_filename"]
            new = out_dir / row["new_filename"]

            if not original.exists():
                print(f"[WARN] Missing source file: {original}")
                n_missing += 1
                continue

            shutil.copy2(original, new)
            n_copied += 1

    # Copy the CSV itself as manifest.csv into the output directory
    manifest_dst = out_dir / "manifest.csv"
    shutil.copy2(csv_path, manifest_dst)

    print(f"\nDone.")
    print(f"  Copied:  {n_copied}")
    print(f"  Missing: {n_missing}")
    print(f"  Output:  {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap and copy map files based on a CSV manifest."
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        required=True,
        help="Directory containing the original map files.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to the CSV manifest with columns "
        "'new_filename' and 'original_filename'.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        required=True,
        help="Base output directory. A subdirectory named after the CSV "
        "(without extension) will be created inside it.",
    )
    args = parser.parse_args()

    remap_files(
        source_path=args.source_path,
        csv_path=args.csv_path,
        target_path=args.target_path,
    )


if __name__ == "__main__":
    main()
