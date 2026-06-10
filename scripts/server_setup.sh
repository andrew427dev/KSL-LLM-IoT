#!/usr/bin/env bash
# GPU 학습 서버 1회 환경 구축 — 서버에서 실행한다.
#   bash scripts/server_setup.sh
# 전제: Python 3.10 (서버 시스템), 디스크 여유 ≥8GB.
# mediapipe·opencv는 설치하지 않는다 — 서버는 키포인트 JSON 변환·학습만 수행.
set -euo pipefail

VENV="${VENV:-$HOME/ksl-venv}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[Setup] venv 생성: $VENV"
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip -q

echo "[Setup] 패키지 설치 (tensorflow GPU 포함, 수 분 소요)..."
# TF 2.16+(Keras 3)는 LSTM의 TFLite 변환이 MLIR 오류로 실패 — Keras 2 내장 2.15 고정
"$VENV/bin/pip" install -q \
    "tensorflow[and-cuda]==2.15.1" \
    "numpy<2" \
    scikit-learn \
    pandas \
    matplotlib \
    tqdm \
    python-dotenv

echo "[Setup] GPU 인식 확인:"
"$VENV/bin/python" - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("  TF", tf.__version__, "| GPU:", gpus if gpus else "(없음 — CPU로 동작)")
EOF

echo "[Setup] 완료. 학습 실행: bash scripts/run_training.sh <dataset_root>"
