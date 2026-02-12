#!/usr/bin/env bash
set -euo pipefail

python GenerateNormalizedDataset4_Xgaze.py \
  --mode remote \
  --profile eth_precise \
  --config configs/benchmark_config_ubuntu_gpu.yaml \
  --mediapipe-backend tasks \
  --mediapipe-delegate gpu \
  --mediapipe-task-model models/face_landmarker.task \
  --annotation-subdir annotation_train \
  --train-subdir train \
  --base-url "https://dataset.ait.ethz.ch/downloads/T3fODqLSS1/eth-xgaze/raw/data/" \
  --output-csv "ETH-GAZE DATASET/processed/training_xgaze_dataset_landmarks_with_gaze.csv" \
  --resume \
  "$@"
