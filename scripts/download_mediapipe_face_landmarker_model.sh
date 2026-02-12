#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
OUT_PATH="models/face_landmarker.task"

mkdir -p "$(dirname "$OUT_PATH")"

if command -v curl >/dev/null 2>&1; then
  curl -fL "$MODEL_URL" -o "$OUT_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT_PATH" "$MODEL_URL"
else
  echo "Error: installa curl o wget per scaricare il modello."
  exit 1
fi

echo "Downloaded: $OUT_PATH"
