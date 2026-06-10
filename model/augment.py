"""
augment.py
data/landmarks/ 의 원본 시퀀스를 증강하여 data/augmented/ 에 저장합니다.

Kim et al. (2026) CMC 논문의 증강 기법을 131차원 레이아웃
([LEFT_63 | RIGHT_63 | wrist_vec_3 | presence_2], src/feature_format.py 참조)에
맞게 구현:
  1. 좌우 flip   — 손 내부 x 부호 반전 + wrist_vec y·z 부호 반전 + 슬롯/flag swap.
                   방향 의존 수어는 FLIP_SAFE_LABELS로 opt-in.
  2. 랜덤 노이즈 — Gaussian noise. presence flag·부재 손 슬롯·(한손 부재 시)
                   wrist_vec에는 적용하지 않는다.
  3. 속도 변화   — 단조증가 앵커 기반 비선형 time warp.

Usage:
    python model/augment.py
    python model/augment.py --factor 3   # 원본 1개 → 증강 3개 (기본값)
"""

import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    KSL_LABELS, SEQUENCE_LENGTH, FEATURE_DIM, INPUT_SHAPE,
    LEFT_SLOT_START, RIGHT_SLOT_START, WRIST_VEC_START, PRESENCE_FLAG_START,
)

_PER_HAND_DIM = RIGHT_SLOT_START - LEFT_SLOT_START  # 63

LANDMARKS_DIR = "data/landmarks"
AUGMENTED_DIR = "data/augmented"

# 좌우 flip이 의미를 보존하는 라벨만 등록한다 (opt-in).
# 방향 자체가 의미인 수어(이동·수여 동사)는 flip 시 동일 라벨의
# 기하학적으로 틀린 샘플이 생기므로 제외: 가다, 오다, 주다, 돕다.
# 나머지는 왼손잡이 수어자의 거울상 수행이 동일 의미이므로 안전.
# 1차 학습 confusion matrix에서 flip 기인 혼동이 보이면 재검토.
FLIP_SAFE_LABELS = {
    "나", "당신", "좋다", "싫다", "맞다", "서다", "자다",
    "배고프다", "목마르다", "아프다", "피곤하다",
    "춥다", "덥다", "슬프다", "화나다",
    "행복", "감사", "부탁",
    "밥", "병원", "의사", "엄마", "가족", "친구", "얼마",
    "완료",
}


# ── 증강 함수 ───────────────────────────────────────────

def flip_horizontal(seq: np.ndarray) -> np.ndarray:
    """
    실제 거울 효과. seq shape: (T, FEATURE_DIM).
      1. 손 내부 x 좌표 부호 반전 (양 슬롯의 0,3,...,60 오프셋 컬럼)
      2. wrist_vec = 우손목−좌손목: 거울 후 손이 서로 바뀌므로 빼는 순서가
         반전되고 x는 거울로 한 번 더 반전 — 결과적으로 x 불변, y·z만 부호 반전
      3. LEFT/RIGHT 슬롯 swap + presence flag swap
    """
    mirrored = seq.copy()
    for start in (LEFT_SLOT_START, RIGHT_SLOT_START):
        mirrored[:, start:start + _PER_HAND_DIM][:, 0::3] *= -1
    mirrored[:, WRIST_VEC_START + 1:WRIST_VEC_START + 3] *= -1

    out = mirrored.copy()
    out[:, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND_DIM] = \
        mirrored[:, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND_DIM]
    out[:, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND_DIM] = \
        mirrored[:, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND_DIM]
    out[:, PRESENCE_FLAG_START] = mirrored[:, PRESENCE_FLAG_START + 1]
    out[:, PRESENCE_FLAG_START + 1] = mirrored[:, PRESENCE_FLAG_START]
    return out


def add_noise(seq: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Gaussian noise 추가. 단 다음 영역은 마스킹한다:
      - presence flag 차원 (이산값 보존)
      - presence=0인 손 슬롯 (존재하지 않는 손에 가짜 신호 방지)
      - 한쪽 손이라도 부재한 프레임의 wrist_vec (정의상 0이어야 함)
    """
    noise = np.random.normal(0, std, seq.shape).astype(np.float32)

    noise[:, PRESENCE_FLAG_START:] = 0.0

    left_absent = seq[:, PRESENCE_FLAG_START] == 0.0
    right_absent = seq[:, PRESENCE_FLAG_START + 1] == 0.0
    noise[left_absent, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND_DIM] = 0.0
    noise[right_absent, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND_DIM] = 0.0
    noise[left_absent | right_absent, WRIST_VEC_START:WRIST_VEC_START + 3] = 0.0

    return seq + noise


def z_jitter(seq: np.ndarray, scale_range=(0.6, 1.4), std: float = 0.05) -> np.ndarray:
    """
    z축 도메인 갭 증강. 학습 데이터(AI Hub)는 멀티뷰 3D 복원의 정밀한 z,
    런타임(MediaPipe)은 단안 추정 z로 스케일·노이즈 특성이 다르다
    (동일 영상 비교 실측 상관 0.577 — tools/verify_aihub_alignment.py).
    z 성분에 랜덤 스케일 + 가우시안 노이즈를 가해 모델의 z 의존도를 낮춘다.

    presence 불변식 유지: 부재 손 슬롯(값 0)은 스케일에 불변이고,
    노이즈는 존재하는 손 슬롯과 양손 존재 프레임의 wrist_vec z에만 가한다.
    """
    out = seq.copy()
    s = np.random.uniform(*scale_range)

    left_present = seq[:, PRESENCE_FLAG_START] == 1.0
    right_present = seq[:, PRESENCE_FLAG_START + 1] == 1.0
    both_present = left_present & right_present

    # 손 슬롯 z: 인덱스 2, 5, ..., 123 (양손 42개)
    z_cols = np.arange(2, WRIST_VEC_START, 3)
    out[:, z_cols] *= s
    noise = np.random.normal(0, std, (len(seq), len(z_cols))).astype(np.float32)
    half = len(z_cols) // 2
    noise[~left_present, :half] = 0.0
    noise[~right_present, half:] = 0.0
    out[:, z_cols] += noise

    # 손목간 벡터 z (인덱스 WRIST_VEC_START+2) — 양손 존재 프레임만
    wz = WRIST_VEC_START + 2
    out[:, wz] *= s
    out[both_present, wz] += np.random.normal(
        0, std, int(both_present.sum())
    ).astype(np.float32)

    return out


def time_warp(seq: np.ndarray, max_jitter: float = 0.15) -> np.ndarray:
    """
    단조증가 랜덤 앵커 기반 비선형 시간 왜곡 — 구간별로 빠르고 느린
    수행 속도 변화를 시뮬레이션한다. 시작·끝 프레임은 보존.

    presence flag는 보간 후 0/1로 재이산화하고, flag=0인 손 슬롯은
    feature_format 불변식(부재 손 = zero)에 맞게 0으로 되돌린다.
    """
    n = len(seq)
    n_anchors = 5
    anchor_pos = np.linspace(0, n - 1, n_anchors)
    jitter = np.random.uniform(-max_jitter, max_jitter, n_anchors) * (n / n_anchors)
    jitter[0] = jitter[-1] = 0.0  # 경계 보존
    warped_anchor = np.maximum.accumulate(np.clip(anchor_pos + jitter, 0, n - 1))
    sample_times = np.interp(np.arange(n), anchor_pos, warped_anchor)

    lo = np.floor(sample_times).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (sample_times - lo)[:, None].astype(np.float32)
    out = seq[lo] * (1.0 - frac) + seq[hi] * frac

    # presence flag 재이산화 + 불변식 복원
    flags = np.round(out[:, PRESENCE_FLAG_START:PRESENCE_FLAG_START + 2])
    out[:, PRESENCE_FLAG_START:PRESENCE_FLAG_START + 2] = flags
    left_absent = flags[:, 0] == 0.0
    right_absent = flags[:, 1] == 0.0
    out[left_absent, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND_DIM] = 0.0
    out[right_absent, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND_DIM] = 0.0
    out[left_absent | right_absent, WRIST_VEC_START:WRIST_VEC_START + 3] = 0.0

    return out.astype(np.float32)


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

    flip_ok = label in FLIP_SAFE_LABELS
    saved = 0
    aug_idx = 0

    for fname in files:
        path = os.path.join(src_dir, fname)
        try:
            seq = np.loadtxt(path, delimiter=",").astype(np.float32)
        except Exception as e:
            print(f"  [Error] {fname}: {e}")
            continue

        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            continue

        # 각 원본 샘플에 대해 factor개 증강 생성.
        # flip은 FLIP_SAFE_LABELS에 등록된 라벨만 — 그 외에는 time_warp+noise로 대체.
        # z_jitter는 모든 증강에 공통 적용 (z 도메인 갭은 변형과 무관하게 존재).
        for _ in range(factor):
            std = np.random.uniform(0.005, 0.02)
            if flip_ok:
                augmentations = [
                    flip_horizontal(seq),
                    add_noise(seq, std=std),
                    time_warp(seq),
                    add_noise(flip_horizontal(seq), std=0.01),
                ]
            else:
                augmentations = [
                    time_warp(seq),
                    add_noise(seq, std=std),
                    time_warp(add_noise(seq, std=0.01)),
                    add_noise(time_warp(seq), std=0.01),
                ]
            chosen = z_jitter(augmentations[aug_idx % len(augmentations)])
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
