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

# 학습에 사용된 코드 버전을 로그에 남긴다 — 체인 pull 무음 실패(HANDOFF §2)
# 같은 사고에서 어떤 코드로 학습됐는지 역추적하는 근거.
echo "=== 코드 버전: $(git -c safe.directory="$PWD" log --oneline -1 2>/dev/null || echo 'git 정보 없음') ==="

echo "=== [1/5] 매칭 스캔 ==="
"$PY" convert_aihub.py --dataset "$DATASET" --scan --exact 2>&1 | tee "$ART/scan.log"

echo "=== [2/5] 키포인트 → CSV 변환 (stride=$STRIDE) ==="
# rm 가드 (PR #9 리뷰 #7): 데이터셋 경로에 키포인트가 없으면 기존
# data/landmarks(git 미추적 — 로컬 수집분 포함 가능)를 파괴하기 전에 중단한다.
if ! find "$DATASET" -maxdepth 4 -type d -name "NIA_SL_WORD*" 2>/dev/null | grep -q .; then
    echo "ERROR: '$DATASET' 하위에 NIA_SL_WORD* 키포인트 디렉터리가 없다 —" >&2
    echo "       데이터셋 경로 오류로 판단, 기존 data/landmarks 보존을 위해 중단한다." >&2
    exit 1
fi
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
