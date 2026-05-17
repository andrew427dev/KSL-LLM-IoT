"""
augment.py
data/landmarks/ 의 원본 시퀀스를 증강하여 data/augmented/ 에 저장합니다.

Kim et al. (2026) CMC 논문의 증강 기법 구현:
  1. 좌우 flip  — x 좌표 부호 반전
  2. 랜덤 노이즈 — Gaussian noise 추가
  3. 속도 변화   — 시퀀스 리샘플링 (빠르게/느리게)

Usage:
    python model/augment.py
    python model/augment.py --factor 3   # 원본 1개 → 증강 3개 (기본값)
"""

import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import KSL_LABELS, SEQUENCE_LENGTH, INPUT_SHAPE

INPUT_DIM = INPUT_SHAPE[1]  # 126 (2 hands × 21 × 3)

LANDMARKS_DIR = "data/landmarks"
AUGMENTED_DIR = "data/augmented"


# ── 증강 함수 ───────────────────────────────────────────

def flip_horizontal(seq: np.ndarray) -> np.ndarray:
    """
    실제 거울 효과 = (1) 모든 x 좌표 부호 반전 + (2) LEFT/RIGHT 손 블록 swap.
    seq shape: (T, 126) = [LEFT_63 | RIGHT_63].
    x 컬럼은 0, 3, 6, ..., 123 — half 경계와 무관하게 동일 규칙.
    """
    augmented = seq.copy()
    augmented[:, 0::3] *= -1
    n_half = seq.shape[1] // 2  # 63
    return np.concatenate([augmented[:, n_half:], augmented[:, :n_half]], axis=1)


def add_noise(seq: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Gaussian noise 추가."""
    noise = np.random.normal(0, std, seq.shape).astype(np.float32)
    return seq + noise


def time_warp(seq: np.ndarray, speed_factor: float = None) -> np.ndarray:
    """
    시퀀스 속도 변화 (리샘플링).
    speed_factor > 1 : 빠른 수화 (프레임 압축)
    speed_factor < 1 : 느린 수화 (프레임 확장)
    """
    if speed_factor is None:
        speed_factor = np.random.uniform(0.7, 1.3)

    n = SEQUENCE_LENGTH
    original_idx = np.linspace(0, n - 1, int(n * speed_factor))
    original_idx = np.clip(original_idx, 0, n - 1)

    warped = np.array([
        seq[int(i)] * (1 - (i % 1)) + seq[min(int(i) + 1, n - 1)] * (i % 1)
        for i in original_idx
    ], dtype=np.float32)

    # SEQUENCE_LENGTH로 다시 리샘플
    target_idx = np.linspace(0, len(warped) - 1, n)
    result = np.array([
        warped[int(i)] * (1 - (i % 1)) + warped[min(int(i) + 1, len(warped) - 1)] * (i % 1)
        for i in target_idx
    ], dtype=np.float32)

    return result


# ── 메인 증강 로직 ──────────────────────────────────────

def augment_label(label: str, factor: int):
    """단어 하나의 전체 샘플을 증강."""
    src_dir = os.path.join(LANDMARKS_DIR, label)
    dst_dir = os.path.join(AUGMENTED_DIR, label)

    if not os.path.exists(src_dir):
        print(f"  [Skip] {label} — 원본 없음")
        return 0

    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".csv")])
    if not files:
        print(f"  [Skip] {label} — CSV 파일 없음")
        return 0

    saved = 0
    aug_idx = 0

    for fname in files:
        path = os.path.join(src_dir, fname)
        try:
            seq = np.loadtxt(path, delimiter=",").astype(np.float32)
        except Exception as e:
            print(f"  [Error] {fname}: {e}")
            continue

        if seq.shape != (SEQUENCE_LENGTH, INPUT_DIM):
            continue

        # 각 원본 샘플에 대해 factor개 증강 생성
        for _ in range(factor):
            augmentations = [
                flip_horizontal(seq),
                add_noise(seq, std=np.random.uniform(0.005, 0.02)),
                time_warp(seq),
                add_noise(flip_horizontal(seq), std=0.01),  # flip + noise 조합
            ]
            chosen = augmentations[aug_idx % len(augmentations)]
            aug_idx += 1

            out_path = os.path.join(dst_dir, f"aug_{saved:05d}.csv")
            np.savetxt(out_path, chosen, delimiter=",")
            saved += 1

    return saved


def augment_all(factor: int = 3):
    print(f"[Augment] 증강 시작 (원본 1개 → 증강 {factor}개)")
    print(f"  원본: {LANDMARKS_DIR}")
    print(f"  대상: {AUGMENTED_DIR}\n")

    total = 0
    for label in KSL_LABELS:
        n = augment_label(label, factor)
        if n > 0:
            print(f"  {label}: +{n}개")
        total += n

    print(f"\n[Augment] 완료. 총 {total}개 증강 샘플 저장.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=3,
                        help="원본 1개당 생성할 증강 샘플 수 (기본: 3)")
    args = parser.parse_args()
    augment_all(args.factor)
