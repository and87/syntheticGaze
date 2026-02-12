#!/usr/bin/env bash
set -euo pipefail

# 1) Create shards first (if needed):
# python scripts/make_subject_shards_from_remote.py --annotation-subdir annotation_train --num-shards 8
#
# 2) Launch:
# ./run_xgaze_remote_train_parallel_8.sh
#
# 3) Merge parts:
# python scripts/merge_csv_parts.py --input-glob "ETH-GAZE DATASET/processed/parallel_train/part_*.csv" \
#   --output-csv "ETH-GAZE DATASET/processed/training_xgaze_dataset_landmarks_with_gaze.csv"

NUM_WORKERS="${NUM_WORKERS:-8}"
SHARD_DIR="${SHARD_DIR:-ETH-GAZE DATASET/processed/subject_shards_train}"
OUT_DIR="${OUT_DIR:-ETH-GAZE DATASET/processed/parallel_train}"
LOG_DIR="${LOG_DIR:-ETH-GAZE DATASET/processed/parallel_train_logs}"
BASE_URL="${BASE_URL:-https://dataset.ait.ethz.ch/downloads/T3fODqLSS1/eth-xgaze/raw/data/}"
ROTATE_CAMS="${ROTATE_CAMS:-3,6,13}"
ANNOTATION_SUBDIR="${ANNOTATION_SUBDIR:-annotation_train}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train}"

if [[ "${ANNOTATION_SUBDIR}" == "annotation_train" ]]; then
  MERGED_OUTPUT_DEFAULT="ETH-GAZE DATASET/processed/training_xgaze_dataset_landmarks_with_gaze.csv"
else
  MERGED_OUTPUT_DEFAULT="ETH-GAZE DATASET/processed/test_xgaze_dataset_landmarks.csv"
fi
MERGED_OUTPUT="${MERGED_OUTPUT:-$MERGED_OUTPUT_DEFAULT}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
  shard_file=$(printf "%s/shard_%02d.txt" "$SHARD_DIR" "$i")
  out_csv=$(printf "%s/part_%02d.csv" "$OUT_DIR" "$i")
  log_file=$(printf "%s/part_%02d.log" "$LOG_DIR" "$i")

  if [[ ! -f "$shard_file" ]]; then
    echo "Missing shard file: $shard_file"
    exit 1
  fi

  gpu_id=$((i % 8))
  echo "Starting worker $i (GPU=$gpu_id) -> $out_csv"
  CUDA_VISIBLE_DEVICES="$gpu_id" nohup python GenerateNormalizedDataset4_Xgaze.py \
    --mode remote \
    --profile eth_precise \
    --mediapipe-backend tasks \
    --mediapipe-delegate gpu \
    --mediapipe-task-model models/face_landmarker.task \
    --annotation-subdir "$ANNOTATION_SUBDIR" \
    --train-subdir "$TRAIN_SUBDIR" \
    --base-url "$BASE_URL" \
    --rotate-cams "$ROTATE_CAMS" \
    --subjects-file "$shard_file" \
    --output-csv "$out_csv" \
    --resume \
    > "$log_file" 2>&1 &
done

echo "Workers started in background."
echo "Check progress with: tail -f \"$LOG_DIR\"/part_00.log"
echo "After completion merge with:"
echo "python scripts/merge_csv_parts.py --input-glob \"$OUT_DIR/part_*.csv\" --output-csv \"$MERGED_OUTPUT\""
