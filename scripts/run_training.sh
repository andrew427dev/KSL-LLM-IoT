#!/usr/bin/env bash
# 서버 원커맨드 학습 파이프라인 — 서버의 저장소 디렉터리에서 실행한다.
#   bash scripts/run_training.sh <dataset_root> [stride] [angles...]
# 예:
#   bash scripts/run_training.sh /mnt/data/ksl/aihub_full   # 전체 각도, stride 15
#   bash scripts/run_training.sh /mnt/data/ksl/aihub_full 10 F D  # 정면+위, stride 10
# 어휘 오버라이드(파이프라인 검증 등): KSL_LABELS_OVERRIDE="단어1,단어2" 환경변수.
# 산출물: /mnt/data/ksl/ksl-artifacts/ (ksl_model.tflite, confusion_matrix.png, 로그)
set -euo pipefail

DATASET="${1:?usage: run_training.sh <dataset_root> [stride] [angles...]}"
STRIDE="${2:-15}"
shift $(( $# > 2 ? 2 : $# )) || true
ANGLES=("$@")

VENV="${VENV:-/mnt/data/ksl/ksl-venv}"
PY="$VENV/bin/python"
ART="${ART:-/mnt/data/ksl/ksl-artifacts}"
mkdir -p "$ART"

echo "=== [1/5] 매칭 스캔 ==="
"$PY" convert_aihub.py --dataset "$DATASET" --scan --exact 2>&1 | tee "$ART/scan.log"

echo "=== [2/5] 키포인트 → CSV 변환 (stride=$STRIDE) ==="
rm -rf data/landmarks data/augmented
if [ ${#ANGLES[@]} -gt 0 ]; then
    "$PY" convert_aihub.py --dataset "$DATASET" --stride "$STRIDE" --exact --angles "${ANGLES[@]}"
else
    "$PY" convert_aihub.py --dataset "$DATASET" --stride "$STRIDE" --exact
fi

echo "=== [3/5] 증강 (factor 3) ==="
"$PY" model/augment.py --factor 3

echo "=== [4/5] LSTM 학습 ==="
"$PY" model/train.py 2>&1 | tee "$ART/train.log"

echo "=== [5/5] 평가 ==="
"$PY" model/evaluate.py 2>&1 | tee "$ART/evaluate.log"

cp model/ksl_model.tflite "$ART/" 2>/dev/null || echo "(tflite 없음 — train.log 확인)"
cp model/confusion_matrix.png "$ART/" 2>/dev/null || true
cp model/best_model.keras "$ART/" 2>/dev/null || true

echo
echo "=== 완료. 산출물: $ART ==="
ls -la "$ART"
