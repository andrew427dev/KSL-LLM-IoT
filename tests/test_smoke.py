"""
test_smoke.py
경량 정적 검증 — mediapipe / tensorflow / tflite-runtime 없이 실행 가능.

CI에서 매 PR/push마다 돌려:
  - 설정 상수(INPUT_SHAPE / NUM_HANDS 등) 일관성
  - augment.flip_horizontal의 거울 효과 수치 회귀
  - hand_tracker.py 소스에 핵심 분기(score 임계값·라벨 충돌 처리·zero-pad) 존재
  - 핵심 모듈에 단일 손 차원(63) 하드코딩 잔재 부재

를 검증한다. mediapipe/tflite 휠을 받지 않으므로 RPi/CI 모두 빠르다.

로컬 실행:
    python tests/test_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config.settings import (
    INPUT_SHAPE, NUM_HANDS, NUM_LANDMARKS, NUM_AXES, SEQUENCE_LENGTH,
)


def test_settings_consistency():
    """INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_HANDS * NUM_LANDMARKS * NUM_AXES)."""
    expected = (SEQUENCE_LENGTH, NUM_HANDS * NUM_LANDMARKS * NUM_AXES)
    assert INPUT_SHAPE == expected, f"INPUT_SHAPE={INPUT_SHAPE} != {expected}"
    assert INPUT_SHAPE == (30, 126), f"INPUT_SHAPE={INPUT_SHAPE}, expected (30, 126)"
    print(f"[OK] settings: INPUT_SHAPE={INPUT_SHAPE}, NUM_HANDS={NUM_HANDS}")


def test_augment_flip_horizontal():
    """거울 효과 = x 부호 반전 + LEFT/RIGHT 블록 swap."""
    from model.augment import flip_horizontal

    seq = np.zeros((SEQUENCE_LENGTH, INPUT_SHAPE[1]), dtype=np.float32)
    seq[:, 0:63] = 1.0   # LEFT (x=1, y=1, z=1)
    seq[:, 63:126] = 2.0 # RIGHT (x=2, y=2, z=2)
    flipped = flip_horizontal(seq)

    # 결과 LEFT 슬롯에는 원래 RIGHT(2.0)가 들어가되 x는 부호 반전 (-2.0).
    assert flipped[0, 0] == -2.0, f"flipped[0,0]={flipped[0,0]}, expected -2.0"
    assert flipped[0, 1] == 2.0,  f"flipped[0,1]={flipped[0,1]}, expected 2.0 (y unchanged)"
    # 결과 RIGHT 슬롯에는 원래 LEFT(1.0)가 들어가되 x는 부호 반전 (-1.0).
    assert flipped[0, 63] == -1.0, f"flipped[0,63]={flipped[0,63]}, expected -1.0"
    assert flipped[0, 64] == 1.0,  f"flipped[0,64]={flipped[0,64]}, expected 1.0 (y unchanged)"
    print("[OK] augment.flip_horizontal: x-flip + LEFT/RIGHT swap")


def test_hand_tracker_source_markers():
    """hand_tracker.py 소스에 양손/충돌 처리 핵심 분기가 존재."""
    path = os.path.join(os.path.dirname(__file__), "..", "src", "hand_tracker.py")
    src = open(path, encoding="utf-8").read()
    required = [
        "max_num_hands=NUM_HANDS",
        "HANDEDNESS_SCORE_THRESHOLD",
        "multi_handedness",
        "np.concatenate([left, right])",
        "handedness conflict",  # 충돌 warning 메시지
    ]
    for needle in required:
        assert needle in src, f"hand_tracker.py missing marker: {needle!r}"
    print("[OK] hand_tracker.py: 양손 + score 임계값 + 충돌 처리 분기 존재")


def test_no_single_hand_hardcoding():
    """핵심 모듈에 (30, 63) / np.zeros(63) 등 단일 손 차원 하드코딩이 없어야 한다."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad_patterns = [
        "shape == (30, 63)",
        "shape == (SEQUENCE_LENGTH, 63)",
        "np.zeros(63",
        "np.random.rand(63)",
    ]
    files = [
        "model/train.py",
        "model/augment.py",
        "tests/test_classifier.py",
        "src/classifier.py",
    ]
    for rel in files:
        s = open(os.path.join(repo_root, rel), encoding="utf-8").read()
        for p in bad_patterns:
            assert p not in s, f"{rel}: 단일 손 차원 잔재 발견 {p!r}"
    print("[OK] no single-hand (63) hardcoding in core modules")


if __name__ == "__main__":
    test_settings_consistency()
    test_augment_flip_horizontal()
    test_hand_tracker_source_markers()
    test_no_single_hand_hardcoding()
    print("\n=== smoke test PASS ===")
