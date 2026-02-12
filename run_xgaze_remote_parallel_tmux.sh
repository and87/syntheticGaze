#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher with tmux. Keeps workers alive if SSH disconnects.
#
# Usage:
#   1) source env/bin/activate
#   2) python scripts/make_subject_shards_from_remote.py --annotation-subdir annotation_train --num-shards 8 --out-dir "ETH-GAZE DATASET/processed/subject_shards_train"
#   3) ./run_xgaze_remote_parallel_tmux.sh
#   4) tmux attach -t xgaze_parallel
#
# Tunables via env vars:
#   SESSION, NUM_WORKERS, SHARD_DIR, OUT_DIR, LOG_DIR, BASE_URL,
#   ROTATE_CAMS, ANNOTATION_SUBDIR, TRAIN_SUBDIR, PYTHON_BIN

SESSION="${SESSION:-xgaze_parallel}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHARD_DIR="${SHARD_DIR:-ETH-GAZE DATASET/processed/subject_shards_train}"
OUT_DIR="${OUT_DIR:-ETH-GAZE DATASET/processed/parallel_train}"
LOG_DIR="${LOG_DIR:-ETH-GAZE DATASET/processed/parallel_train_logs}"
BASE_URL="${BASE_URL:-https://dataset.ait.ethz.ch/downloads/T3fODqLSS1/eth-xgaze/raw/data/}"
ROTATE_CAMS="${ROTATE_CAMS:-3,6,13}"
ANNOTATION_SUBDIR="${ANNOTATION_SUBDIR:-annotation_train}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux non trovato. Installa tmux e riprova."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Error: sessione tmux '$SESSION' gia' esistente."
  echo "Chiudila con: tmux kill-session -t $SESSION"
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
  shard_file=$(printf "%s/shard_%02d.txt" "$SHARD_DIR" "$i")
  if [[ ! -f "$shard_file" ]]; then
    echo "Missing shard file: $shard_file"
    exit 1
  fi
done

for i in $(seq 0 $((NUM_WORKERS - 1))); do
  window=$(printf "w%02d" "$i")
  shard_file=$(printf "%s/shard_%02d.txt" "$SHARD_DIR" "$i")
  out_csv=$(printf "%s/part_%02d.csv" "$OUT_DIR" "$i")
  log_file=$(printf "%s/part_%02d.log" "$LOG_DIR" "$i")
  gpu_id=$((i % 8))

  if [[ "$i" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$window"
  else
    tmux new-window -t "$SESSION" -n "$window"
  fi

  cmd="cd \"$PWD\" && CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_BIN GenerateNormalizedDataset4_Xgaze.py \
--mode remote \
--profile eth_precise \
--mediapipe-backend tasks \
--mediapipe-delegate gpu \
--mediapipe-task-model models/face_landmarker.task \
--annotation-subdir \"$ANNOTATION_SUBDIR\" \
--train-subdir \"$TRAIN_SUBDIR\" \
--base-url \"$BASE_URL\" \
--rotate-cams \"$ROTATE_CAMS\" \
--subjects-file \"$shard_file\" \
--output-csv \"$out_csv\" \
--resume \
> \"$log_file\" 2>&1"

  tmux send-keys -t "$SESSION:$window" "$cmd" C-m
done

tmux new-window -t "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" "cd \"$PWD\" && ls -lh \"$LOG_DIR\" && tail -f \"$LOG_DIR\"/part_00.log" C-m

echo "Sessione tmux avviata: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Logs: $LOG_DIR"
echo "Output CSV parts: $OUT_DIR/part_*.csv"
echo "Merge finale:"
echo "python scripts/merge_csv_parts.py --input-glob \"$OUT_DIR/part_*.csv\" --output-csv \"ETH-GAZE DATASET/processed/training_xgaze_dataset_landmarks_with_gaze.csv\""

