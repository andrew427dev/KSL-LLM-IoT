#!/usr/bin/env bash
# AI Hub 데이터셋을 서버로 전송한다 — PC(WSL)에서 실행.
#   bash scripts/upload_dataset.sh <dataset_root> [remote_dir]
# 원천 MP4는 변환에 불필요하므로 제외하고 키포인트·형태소 JSON만 tar 스트림으로
# 전송한다 (중간 파일 없음, JSON 압축률 높음).
set -euo pipefail

DATASET="${1:?usage: upload_dataset.sh <dataset_root> [remote_dir]}"
REMOTE_DIR="${2:-/mnt/data/ksl/aihub_dataset}"
HOST="${KSL_GPU_HOST:-root@cscloud.gpu3.hufs.ac.kr}"
PORT="${KSL_GPU_PORT:-30007}"

echo "[Upload] $DATASET → $HOST:$REMOTE_DIR (MP4 제외)"
tar -czf - \
    --exclude='*.mp4' --exclude='*.MP4' --exclude='*.mov' --exclude='*.avi' \
    -C "$DATASET" . \
  | ssh -p "$PORT" "$HOST" "mkdir -p '$REMOTE_DIR' && tar -xzf - -C '$REMOTE_DIR'"

echo "[Upload] 완료. 서버에서: bash scripts/run_training.sh $REMOTE_DIR"
