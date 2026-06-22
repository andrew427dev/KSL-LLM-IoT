#!/usr/bin/env bash
# 서버 재학습 체인 표준 — 코드 동기화·검증·학습을 한 단위로 묶는다.
#   bash scripts/chain_train.sh <dataset_root> [wait_pid]
# 예:
#   bash scripts/chain_train.sh /mnt/data/ksl/aihub_full          # 즉시 학습
#   bash scripts/chain_train.sh /mnt/data/ksl/aihub_full 1821500  # PID 종료 대기 후
#
# 배경 (함정 2-U): deploy_code.sh의 tar 배포가 작업트리를 dirty로
# 만들어 `git pull`이 무음 실패 → 구버전 코드로 학습되는 사고가 있었다.
# 본 스크립트는 ① dirty 트리에 면역인 fetch+reset 동기화, ② 학습 직전
# 코드 표지 검증, ③ set -e로 어느 단계든 실패 시 즉시 중단을 보장한다.
# 주의: reset --hard는 서버 작업트리의 미커밋 변경을 폐기한다 — 서버에서
# 코드를 직접 수정하지 않는다는 운용 전제(코드 변경은 PC→push→서버 pull).
set -euo pipefail

DATASET="${1:?usage: chain_train.sh <dataset_root> [wait_pid]}"
WAIT_PID="${2:-}"
BRANCH="${BRANCH:-develop}"

if [ -n "$WAIT_PID" ]; then
    echo "[Chain] PID $WAIT_PID 종료 대기..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 180; done
    echo "[Chain] PID $WAIT_PID 종료 감지."
fi

cd "$(dirname "$0")/.."
GIT=(git -c safe.directory="$PWD")

echo "[Chain] origin/$BRANCH 동기화 (fetch + reset --hard — dirty 트리 면역)"
"${GIT[@]}" fetch origin
"${GIT[@]}" reset --hard "origin/$BRANCH"
echo "[Chain] 코드 버전: $("${GIT[@]}" log --oneline -1)"

# 코드 표지 검증 — 핵심 수정이 실제로 작업트리에 존재하는지 학습 전에 확인
grep -q "label_smoothing" model/train.py
grep -q "z_jitter" model/augment.py
grep -q "split_holdout" model/train.py
echo "[Chain] 코드 표지 검증 통과 (label_smoothing·z_jitter·split_holdout)"

bash scripts/run_training.sh "$DATASET"
echo "[Chain] 재학습 종료 (exit 0)"
