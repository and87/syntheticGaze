#!/usr/bin/env bash
set -euo pipefail

python GenerateNormalizedDataset4_Xgaze.py \
  --mode remote \
  --profile eth_precise \
  --annotation-subdir annotation_test \
  --train-subdir test \
  --output-csv "ETH-GAZE DATASET/processed/test_xgaze_dataset_landmarks.csv" \
  --resume \
  "$@"
