import argparse
import csv
import math
import os
from typing import Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert gaze label convention in an existing semicolon CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--compat-preset",
        choices=["none", "legacy_xgaze_v1"],
        default="none",
        help=(
            "legacy_xgaze_v1 applies normalized-space compatibility transform: "
            "invert + swap_xy."
        ),
    )
    parser.add_argument(
        "--gaze-label-sign",
        choices=["as_is", "invert"],
        default="as_is",
        help="Direction transform to apply to gaze vectors.",
    )
    parser.add_argument(
        "--gaze-label-swap-xy",
        action="store_true",
        help="Swap x/y components of gaze vectors.",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter (default: ';').",
    )
    parser.add_argument(
        "--keep-yaw-pitch",
        action="store_true",
        help="Do not recompute gaze_yaw and gaze_pitch columns.",
    )
    return parser.parse_args()


def vector_to_pitch_yaw(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    norm = np.linalg.norm([x, y, z])
    if norm > 0:
        x, y, z = x / norm, y / norm, z / norm
    pitch = float(np.arcsin(np.clip(y, -1.0, 1.0)))
    yaw = float(np.arctan2(x, -z))
    return yaw, pitch


def parse_gaze_from_row(row: dict) -> Optional[np.ndarray]:
    try:
        x = float(row["gaze_x"])
        y = float(row["gaze_y"])
        z = float(row["gaze_z"])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    vec = np.array([x, y, z], dtype=np.float64)
    n = np.linalg.norm(vec)
    if not math.isfinite(n) or n <= 0:
        return None
    return vec / n


def transform_gaze(vec: np.ndarray, sign: str, swap_xy: bool) -> Optional[np.ndarray]:
    out = np.asarray(vec, dtype=np.float64).reshape(3).copy()
    if sign == "invert":
        out = -out
    elif sign != "as_is":
        raise ValueError(f"Unsupported sign: {sign}")

    if swap_xy:
        out = np.array([out[1], out[0], out[2]], dtype=np.float64)

    n = np.linalg.norm(out)
    if not np.isfinite(n) or n <= 0:
        return None
    return out / n


def main() -> None:
    args = parse_args()

    if os.path.abspath(args.input_csv) == os.path.abspath(args.output_csv):
        raise SystemExit("--input-csv and --output-csv must be different files.")

    if args.compat_preset == "legacy_xgaze_v1":
        args.gaze_label_sign = "invert"
        args.gaze_label_swap_xy = True

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    total_rows = 0
    converted_rows = 0
    skipped_rows = 0

    with open(args.input_csv, "r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src, delimiter=args.delimiter)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header.")

        required = {"gaze_x", "gaze_y", "gaze_z"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise SystemExit(f"Missing required columns: {missing}")

        with open(args.output_csv, "w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, delimiter=args.delimiter)
            writer.writeheader()

            has_yaw_pitch = "gaze_yaw" in reader.fieldnames and "gaze_pitch" in reader.fieldnames

            for row in reader:
                total_rows += 1
                vec = parse_gaze_from_row(row)
                if vec is None:
                    skipped_rows += 1
                    writer.writerow(row)
                    continue

                out = transform_gaze(
                    vec=vec,
                    sign=args.gaze_label_sign,
                    swap_xy=bool(args.gaze_label_swap_xy),
                )
                if out is None:
                    skipped_rows += 1
                    writer.writerow(row)
                    continue

                row["gaze_x"] = f"{out[0]:.16g}"
                row["gaze_y"] = f"{out[1]:.16g}"
                row["gaze_z"] = f"{out[2]:.16g}"

                if has_yaw_pitch and not args.keep_yaw_pitch:
                    yaw, pitch = vector_to_pitch_yaw(out)
                    row["gaze_yaw"] = f"{yaw:.16g}"
                    row["gaze_pitch"] = f"{pitch:.16g}"

                converted_rows += 1
                writer.writerow(row)

    print("=== Conversion complete ===")
    print(f"input_csv:  {args.input_csv}")
    print(f"output_csv: {args.output_csv}")
    print(f"preset:     {args.compat_preset}")
    print(f"sign:       {args.gaze_label_sign}")
    print(f"swap_xy:    {bool(args.gaze_label_swap_xy)}")
    print(f"rows_total: {total_rows}")
    print(f"rows_conv:  {converted_rows}")
    print(f"rows_skip:  {skipped_rows}")


if __name__ == "__main__":
    main()
