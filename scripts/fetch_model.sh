#!/usr/bin/env bash
# 서버 학습 산출물(.tflite, 평가 결과)을 PC로 회수한다 — PC(WSL)에서 실행.
#   bash scripts/fetch_model.sh [local_dir]
set -euo pipefail

LOCAL_DIR="${1:-model}"
HOST="${KSL_GPU_HOST:-root@cscloud.gpu3.hufs.ac.kr}"
PORT="${KSL_GPU_PORT:-30007}"

mkdir -p "$LOCAL_DIR"
scp -P "$PORT" \
    "$HOST:ksl-artifacts/ksl_model.tflite" \
    "$HOST:ksl-artifacts/confusion_matrix.png" \
    "$HOST:ksl-artifacts/train.log" \
    "$HOST:ksl-artifacts/evaluate.log" \
    "$LOCAL_DIR/"

echo "[Fetch] 완료 → $LOCAL_DIR"
ls -la "$LOCAL_DIR" | grep -E "tflite|png|log"
