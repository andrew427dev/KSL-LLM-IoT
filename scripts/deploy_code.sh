#!/usr/bin/env bash
# 저장소 코드를 서버로 동기화한다 — PC(WSL)에서 실행.
#   bash scripts/deploy_code.sh [remote_dir]
# .venv / .git / data / __pycache__ 등 무거운 비코드 디렉터리는 제외한다.
set -euo pipefail

REMOTE_DIR="${1:-/mnt/data/ksl/KSL-LLM-IoT}"
HOST="${KSL_GPU_HOST:-root@cscloud.gpu3.hufs.ac.kr}"
PORT="${KSL_GPU_PORT:-30007}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[Deploy] $REPO_ROOT → $HOST:$REMOTE_DIR"
tar -czf - \
    --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    --exclude='data' --exclude='.omc' --exclude='.remember' \
    --exclude='docs/superpowers' --exclude='*.tflite' --exclude='*.keras' \
    --exclude='.env' \
    -C "$REPO_ROOT" . \
  | ssh -p "$PORT" "$HOST" "mkdir -p '$REMOTE_DIR' && tar -xzf - -C '$REMOTE_DIR'"

echo "[Deploy] 완료."
