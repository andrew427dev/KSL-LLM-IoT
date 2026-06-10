"""
test_augment_two_hands.py
131차원 양손 포맷에 대한 증강(model/augment.py) 검증.
numpy + settings만 필요 — mediapipe/tensorflow 불필요.

로컬 실행:
    python tests/test_augment_two_hands.py
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SEQUENCE_LENGTH, FEATURE_DIM,
    LEFT_SLOT_START, RIGHT_SLOT_START, WRIST_VEC_START, PRESENCE_FLAG_START,
)
from model.augment import flip_horizontal, add_noise, time_warp, z_jitter

_PER_HAND = RIGHT_SLOT_START - LEFT_SLOT_START  # 63


def make_seq(left_val=None, right_val=None, wrist_vec=None):
    """(T, 131) 테스트 시퀀스. left_val/right_val이 None이면 해당 손 부재."""
    seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    if left_val is not None:
        seq[:, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND] = left_val
        seq[:, PRESENCE_FLAG_START] = 1.0
    if right_val is not None:
        seq[:, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND] = right_val
        seq[:, PRESENCE_FLAG_START + 1] = 1.0
    if wrist_vec is not None:
        seq[:, WRIST_VEC_START:WRIST_VEC_START + 3] = wrist_vec
    return seq


# ── T9: flip — 한 손 시퀀스의 슬롯/flag swap ──────────────────

def test_flip_left_only_becomes_right_only():
    seq = make_seq(left_val=1.0)
    flipped = flip_horizontal(seq)
    # 좌손(전부 1.0: x=y=z=1) → 우손 슬롯으로, x만 부호 반전
    assert flipped[0, PRESENCE_FLAG_START] == 0.0
    assert flipped[0, PRESENCE_FLAG_START + 1] == 1.0
    assert np.all(flipped[:, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND] == 0.0)
    right = flipped[:, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND]
    assert np.all(right[:, 0::3] == -1.0), "x는 부호 반전"
    assert np.all(right[:, 1::3] == 1.0), "y는 불변"
    assert np.all(right[:, 2::3] == 1.0), "z는 불변"
    print("[PASS] test_flip_left_only_becomes_right_only")


def test_flip_wrist_vec_sign():
    """flip 시 wrist_vec은 x 불변, y·z 부호 반전 (거울 + 빼는 순서 반전의 합성)."""
    seq = make_seq(left_val=1.0, right_val=2.0, wrist_vec=(4.0, 1.0, 2.0))
    flipped = flip_horizontal(seq)
    np.testing.assert_allclose(
        flipped[0, WRIST_VEC_START:WRIST_VEC_START + 3], [4.0, -1.0, -2.0]
    )
    print("[PASS] test_flip_wrist_vec_sign")


def test_flip_is_involution():
    """flip은 자기 자신의 역연산 — 두 번 적용하면 원본과 일치해야 한다."""
    rng = np.random.default_rng(0)
    seq = make_seq(left_val=0.0, right_val=0.0)
    seq[:, :WRIST_VEC_START + 3] = rng.normal(
        size=(SEQUENCE_LENGTH, WRIST_VEC_START + 3)
    ).astype(np.float32)
    np.testing.assert_allclose(flip_horizontal(flip_horizontal(seq)), seq, atol=1e-6)
    print("[PASS] test_flip_is_involution")


# ── T10: noise 마스킹 ─────────────────────────────────────────

def test_noise_masks_flags_and_absent_slots():
    np.random.seed(1)
    seq = make_seq(left_val=0.5)  # 우손 부재
    noised = add_noise(seq, std=0.05)
    # presence flag 불변
    np.testing.assert_array_equal(
        noised[:, PRESENCE_FLAG_START:], seq[:, PRESENCE_FLAG_START:]
    )
    # 부재 손 슬롯 불변 (zero 유지)
    np.testing.assert_array_equal(
        noised[:, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND],
        seq[:, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND],
    )
    # 한쪽 부재 프레임의 wrist_vec 불변 (zero 유지)
    np.testing.assert_array_equal(
        noised[:, WRIST_VEC_START:WRIST_VEC_START + 3],
        seq[:, WRIST_VEC_START:WRIST_VEC_START + 3],
    )
    # 존재하는 좌손 슬롯에는 노이즈가 들어가야 함
    assert np.any(
        noised[:, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND]
        != seq[:, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND]
    )
    print("[PASS] test_noise_masks_flags_and_absent_slots")


# ── time_warp ─────────────────────────────────────────────────

def test_time_warp_shape_and_boundaries():
    np.random.seed(2)
    seq = make_seq(left_val=0.0, right_val=0.0, wrist_vec=(1.0, 0.0, 0.0))
    # 프레임별로 다른 값을 줘 보간 효과 확인
    ramp = np.linspace(0, 1, SEQUENCE_LENGTH, dtype=np.float32)[:, None]
    seq[:, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND] = ramp

    warped = time_warp(seq)
    assert warped.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert warped.dtype == np.float32
    # 경계 보존: 첫/마지막 프레임은 원본과 동일
    np.testing.assert_allclose(warped[0], seq[0], atol=1e-6)
    np.testing.assert_allclose(warped[-1], seq[-1], atol=1e-6)
    # presence flag는 0/1 이산값 유지
    flags = warped[:, PRESENCE_FLAG_START:]
    assert np.all((flags == 0.0) | (flags == 1.0))
    print("[PASS] test_time_warp_shape_and_boundaries")


def test_time_warp_restores_absent_slot_invariant():
    """보간 과정에서 flag=0 프레임의 손 슬롯·wrist_vec이 0으로 복원되는지."""
    np.random.seed(3)
    seq = make_seq(left_val=0.7)  # 우손 부재 (flag 0)
    warped = time_warp(seq)
    right_absent = warped[:, PRESENCE_FLAG_START + 1] == 0.0
    assert np.all(
        warped[right_absent][:, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND] == 0.0
    )
    assert np.all(warped[right_absent][:, WRIST_VEC_START:WRIST_VEC_START + 3] == 0.0)
    print("[PASS] test_time_warp_restores_absent_slot_invariant")


def test_z_jitter_only_touches_z():
    """z_jitter는 z 성분만 변경 — x·y·flag 불변, 부재 슬롯 z는 0 유지."""
    np.random.seed(4)
    seq = make_seq(left_val=0.5, wrist_vec=None)  # 우손 부재
    out = z_jitter(seq)
    # x(0::3)·y(1::3) 불변 (손 슬롯 구간)
    np.testing.assert_array_equal(
        out[:, 0:WRIST_VEC_START:3], seq[:, 0:WRIST_VEC_START:3])
    np.testing.assert_array_equal(
        out[:, 1:WRIST_VEC_START:3], seq[:, 1:WRIST_VEC_START:3])
    # presence flag 불변
    np.testing.assert_array_equal(
        out[:, PRESENCE_FLAG_START:], seq[:, PRESENCE_FLAG_START:])
    # 존재하는 좌손 z는 변함
    assert np.any(out[:, 2:_PER_HAND:3] != seq[:, 2:_PER_HAND:3])
    # 부재 우손 슬롯 z는 0 유지
    assert np.all(out[:, RIGHT_SLOT_START + 2:WRIST_VEC_START:3] == 0.0)
    # 한쪽 부재 — wrist_vec z도 0 유지
    assert np.all(out[:, WRIST_VEC_START + 2] == 0.0)
    print("[PASS] test_z_jitter_only_touches_z")


if __name__ == "__main__":
    test_flip_left_only_becomes_right_only()
    test_flip_wrist_vec_sign()
    test_flip_is_involution()
    test_noise_masks_flags_and_absent_slots()
    test_time_warp_shape_and_boundaries()
    test_time_warp_restores_absent_slot_invariant()
    test_z_jitter_only_touches_z()
    print("\nAll tests done.")
