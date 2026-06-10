"""
verify_aihub_alignment.py
AI Hub 3D 키포인트와 런타임 MediaPipe 표현 사이의 축 방향·좌우 대응을 실측한다.

원리: AI Hub 데이터셋에는 같은 영상의 원천 MP4와 키포인트 JSON이 모두 있다.
동일 영상에 대해
  (a) 런타임 경로 그대로 — cv2.flip(frame,1) → HandTracker(mirrored=True) → 131벡터
  (b) AI Hub 키포인트 → (후보 축 변환 적용) → feature_format → 131벡터
를 프레임별로 만들고, 손 슬롯 feature의 상관계수를 비교한다.

탐색 공간: 축 부호 8조합(±x, ±y, ±z) × 좌우 슬롯 매핑 2(그대로/swap) = 16.
최적 조합과 축별 상관계수를 출력한다 — 이 결과를 convert_aihub.py의
AIHUB_AXIS_SIGNS / AIHUB_SWAP_LR 상수에 반영한다.

Usage:
    .venv/bin/python tools/verify_aihub_alignment.py \
        --dataset /path/to/aihub/dataset [--videos 3] [--angle F]
"""

import argparse
import glob
import itertools
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config.settings import (
    NUM_LANDMARKS, LEFT_SLOT_START, RIGHT_SLOT_START,
    WRIST_VEC_START, PRESENCE_FLAG_START,
)
from src.feature_format import build_feature_vector
from src.hand_tracker import HandTracker

_PER_HAND = RIGHT_SLOT_START - LEFT_SLOT_START  # 63


def load_aihub_frames(kp_dir):
    """키포인트 디렉터리 → 프레임별 (left(21,3)|None, right(21,3)|None) 목록."""
    frames = []
    for jf in sorted(glob.glob(os.path.join(kp_dir, "*_keypoints.json"))):
        try:
            with open(jf, encoding="utf-8") as fp:
                people = json.load(fp).get("people", {})
        except (json.JSONDecodeError, OSError):
            frames.append((None, None))
            continue
        hands = []
        for key in ("hand_left_keypoints_3d", "hand_right_keypoints_3d"):
            kp = people.get(key, [])
            if not kp or len(kp) < NUM_LANDMARKS * 4:
                hands.append(None)
                continue
            coords = np.array(
                [[kp[i * 4], kp[i * 4 + 1], kp[i * 4 + 2]] for i in range(NUM_LANDMARKS)],
                dtype=np.float32,
            )
            hands.append(coords)
        frames.append(tuple(hands))
    return frames


def runtime_vectors(mp4_path, n_frames, tracker):
    """런타임 경로 그대로 MP4에서 프레임별 131벡터(또는 None)를 추출."""
    cap = cv2.VideoCapture(mp4_path)
    out = []
    while len(out) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # src/main.py:235와 동일
        out.append(tracker.extract_landmarks(frame, mirrored=True))
    cap.release()
    return out


def aihub_vector(left, right, signs, swap_lr):
    """후보 변환(축 부호 + 좌우 swap)을 적용해 AI Hub 손 좌표 → 131벡터."""
    s = np.asarray(signs, dtype=np.float32)
    l = left * s if left is not None else None
    r = right * s if right is not None else None
    if swap_lr:
        l, r = r, l
    return build_feature_vector(l, r)


def collect_pairs(video_pairs, signs, swap_lr):
    """모든 프레임에서 (aihub_slot_feature, runtime_slot_feature) 쌍 수집.
    Returns: 축별 (aihub_vals, runtime_vals) — 손 내부 정규화 좌표만 사용."""
    by_axis = {0: ([], []), 1: ([], []), 2: ([], [])}
    n_used = 0
    for ai_frames, rt_vecs in video_pairs:
        for (left, right), rt in zip(ai_frames, rt_vecs):
            if rt is None:
                continue
            ai = aihub_vector(left, right, signs, swap_lr)
            if ai is None:
                continue
            for start in (LEFT_SLOT_START, RIGHT_SLOT_START):
                flag_idx = PRESENCE_FLAG_START + (0 if start == LEFT_SLOT_START else 1)
                if ai[flag_idx] != 1.0 or rt[flag_idx] != 1.0:
                    continue  # 양쪽 다 잡힌 슬롯만 비교
                a = ai[start:start + _PER_HAND].reshape(NUM_LANDMARKS, 3)
                r = rt[start:start + _PER_HAND].reshape(NUM_LANDMARKS, 3)
                for axis in range(3):
                    by_axis[axis][0].extend(a[:, axis])
                    by_axis[axis][1].extend(r[:, axis])
                n_used += 1
    return by_axis, n_used


def corr(xs, ys):
    if len(xs) < 10:
        return float("nan")
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--videos", type=int, default=3, help="비교할 영상 수")
    parser.add_argument("--angle", default="F", help="카메라 각도 (기본 F=정면)")
    args = parser.parse_args()

    # MP4 ↔ keypoint 디렉터리 짝 찾기
    mp4s = sorted(glob.glob(
        os.path.join(args.dataset, "**", f"*_{args.angle}.mp4"), recursive=True
    ))
    pairs = []
    for mp4 in mp4s:
        base = os.path.basename(mp4)[:-4]
        kp_hits = glob.glob(os.path.join(
            args.dataset, "**", "*keypoint*", base), recursive=True)
        if kp_hits:
            pairs.append((mp4, kp_hits[0]))
        if len(pairs) >= args.videos:
            break

    if not pairs:
        print("[Verify] MP4-키포인트 짝을 찾지 못했다.")
        sys.exit(1)

    print(f"[Verify] 비교 영상 {len(pairs)}개 (각도 {args.angle}):")
    tracker = HandTracker()
    video_pairs = []
    for mp4, kp_dir in pairs:
        ai_frames = load_aihub_frames(kp_dir)
        rt_vecs = runtime_vectors(mp4, len(ai_frames), tracker)
        n = min(len(ai_frames), len(rt_vecs))
        video_pairs.append((ai_frames[:n], rt_vecs[:n]))
        detected = sum(1 for v in rt_vecs if v is not None)
        print(f"  {os.path.basename(mp4)}: keypoint {len(ai_frames)}프레임, "
              f"MediaPipe 검출 {detected}/{len(rt_vecs)}")
    tracker.release()

    # 16개 후보 변환 전수 탐색
    print(f"\n{'signs':>14s} {'swapLR':>7s} {'r_x':>7s} {'r_y':>7s} {'r_z':>7s} "
          f"{'r_mean':>7s} {'pairs':>7s}")
    results = []
    for signs in itertools.product((1, -1), repeat=3):
        for swap_lr in (False, True):
            by_axis, n_used = collect_pairs(video_pairs, signs, swap_lr)
            rs = [corr(*by_axis[a]) for a in range(3)]
            mean_r = float(np.nanmean(rs))
            results.append((mean_r, signs, swap_lr, rs, n_used))
            print(f"{str(signs):>14s} {str(swap_lr):>7s} "
                  f"{rs[0]:>7.3f} {rs[1]:>7.3f} {rs[2]:>7.3f} "
                  f"{mean_r:>7.3f} {n_used:>7d}")

    best = max(results, key=lambda t: (t[0] if not np.isnan(t[0]) else -2))
    mean_r, signs, swap_lr, rs, n_used = best
    print("\n[Verify] 최적 변환:")
    print(f"  AIHUB_AXIS_SIGNS = {tuple(signs)}   # (x, y, z) 부호")
    print(f"  AIHUB_SWAP_LR    = {swap_lr}")
    print(f"  축별 상관계수: x={rs[0]:.3f}  y={rs[1]:.3f}  z={rs[2]:.3f} "
          f"(슬롯-프레임 쌍 {n_used}개)")
    if rs[2] < 0.5:
        print("  주의: z 상관이 낮다 — MediaPipe의 단안 추정 z와 AI Hub의 "
              "멀티뷰 복원 z 간 차이. x·y가 높으면 학습 신호로는 충분하다.")
    if mean_r < 0.6:
        print("  경고: 전체 상관이 낮다 — 변환 적용 전 원인 분석 필요.")


if __name__ == "__main__":
    main()
