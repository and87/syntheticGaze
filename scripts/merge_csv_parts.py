#!/usr/bin/env python3
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser(description="Merge part_*.csv files produced by parallel runs.")
    ap.add_argument(
        "--input-glob",
        default=os.path.join("ETH-GAZE DATASET", "processed", "parallel_train", "part_*.csv"),
    )
    ap.add_argument(
        "--output-csv",
        default=os.path.join("ETH-GAZE DATASET", "processed", "training_xgaze_dataset_landmarks_with_gaze.csv"),
    )
    args = ap.parse_args()

    part_files = sorted(glob.glob(args.input_glob))
    if not part_files:
        raise SystemExit(f"No input files matched: {args.input_glob}")

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = None
    rows_written = 0
    with open(args.output_csv, "w", encoding="utf-8", newline="") as out:
        for idx, path in enumerate(part_files):
            with open(path, "r", encoding="utf-8", newline="") as f:
                lines = f.readlines()
            if not lines:
                continue
            part_header = lines[0]
            if header is None:
                header = part_header
                out.write(header)
            elif part_header.strip() != header.strip():
                raise SystemExit(f"Header mismatch in {path}")
            for line in lines[1:]:
                if line.strip():
                    out.write(line)
                    rows_written += 1
            print(f"Merged: {path}")

    print(f"Done. Wrote {rows_written} rows into {args.output_csv}")


if __name__ == "__main__":
    main()

