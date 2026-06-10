#!/usr/bin/env bash
# AI Hub 수어 데이터셋(103)에서 필요한 파일만 받아 학습용 디렉터리를 구성한다.
# GPU 서버에서 실행. 디스크 절약: 시연자 zip을 1개씩 받아 KSL_LABELS의
# WORD 디렉터리만 선택 해제 후 zip을 즉시 삭제한다.
#
#   bash scripts/server_download_aihub.sh <시연자수 1-16>
#
# 전제: /mnt/data/ksl/.aihub_key 파일에 AI Hub API 키 (chmod 600),
#       데이터셋 103 이용신청 승인 상태.
set -euo pipefail

# 사전 점검 — 컨테이너 기본 이미지에 unzip·curl이 없을 수 있다
for cmd in unzip curl aihubshell; do
    command -v "$cmd" >/dev/null || { echo "ERROR: '$cmd' 미설치 — apt-get install -y unzip curl 후 재실행"; exit 1; }
done

N_SIGNERS="${1:-3}"
KEY="$(cat /mnt/data/ksl/.aihub_key)"
DEST="${DEST:-/mnt/data/ksl/aihub_full}"
WORK="${WORK:-/mnt/data/ksl/aihub_dl}"
mkdir -p "$DEST/라벨링데이터/REAL/WORD" "$DEST/morpheme" "$WORK"

# KSL_LABELS 30단어의 WORD 패키지 45개 (HANDOFF §1.5, --exact 매칭 확정)
WORDS="1157 1353 0738 1191 1278 1385 1174 2317 2318 0943 0944 0946 1345 1149 \
1148 1344 1377 1378 1544 2394 2395 2396 0953 1197 2036 1152 1158 1248 1382 \
0009 1236 1169 1290 1589 2388 2389 2390 1534 1496 0163 1528 1492 1204 0436 0742"

# filekey 매핑 (aihubshell -mode l -datasetkey 103 기준, 2026-06-10)
MORPHEME_KEY=39601
# 시연자 01~16 → real_word_keypoint.zip filekey
KP_KEYS=(39600 39602 39603 39604 39605 39606 39607 39608 39609 39610 39611 39612 39613 39614 39615 39616)

cd "$WORK"

echo "=== [1] 형태소 매핑 (110MB) ==="
if [ ! -e "$DEST/.morpheme_done" ]; then
    aihubshell -mode d -datasetkey 103 -filekey "$MORPHEME_KEY" -aihubapikey "$KEY"
    find . -name "*.zip" -exec unzip -o -q {} -d "$DEST/morpheme" \; -delete
    touch "$DEST/.morpheme_done"
fi
echo "  morpheme JSON: $(find "$DEST/morpheme" -name "*_morpheme.json" | wc -l)개"

echo "=== [2] 시연자별 키포인트 (각 ~11GB 다운로드 → 45 WORD만 해제 → zip 삭제) ==="
for i in $(seq 1 "$N_SIGNERS"); do
    idx=$((i - 1))
    fk="${KP_KEYS[$idx]}"
    marker="$DEST/.signer_${i}_done"
    if [ -e "$marker" ]; then
        echo "  [시연자 $i] 이미 완료 — skip"
        continue
    fi
    echo "  [시연자 $i] filekey=$fk 다운로드 중..."
    df -h / | tail -1
    aihubshell -mode d -datasetkey 103 -filekey "$fk" -aihubapikey "$KEY"

    # aihubshell은 zip을 데이터셋 트리 경로(004.수어영상/...) 하위에 받고
    # 분할(part) 파일은 자동 병합한다 — find로 깊이 무관하게 탐색한다.
    while IFS= read -r -d "" z; do
        echo "  [시연자 $i] $(basename "$z") 에서 45개 WORD 선택 해제..."
        for w in $WORDS; do
            unzip -o -q "$z" "*WORD${w}_*" -d "$DEST/라벨링데이터/REAL/WORD" 2>/dev/null || true
        done
        rm -f "$z"
    done < <(find "$WORK" -name "*keypoint*.zip" -print0)
    touch "$marker"
    echo "  [시연자 $i] 완료. 누적 키포인트 디렉터리: $(find "$DEST/라벨링데이터" -maxdepth 4 -type d -name "NIA_SL_WORD*" | wc -l)개"
done

echo "=== 다운로드 완료 ==="
du -sh "$DEST"
find "$DEST/라벨링데이터" -maxdepth 4 -type d -name "NIA_SL_WORD*" | wc -l
