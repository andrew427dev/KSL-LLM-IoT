"""
test_classifier.py
KSLClassifier 유닛 테스트.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SEQUENCE_LENGTH, INPUT_SHAPE, NO_HAND_RESET_FRAMES

INPUT_DIM = INPUT_SHAPE[1]  # FEATURE_DIM = 131 (양손 126 + wrist_vec 3 + presence 2)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/ksl_model.tflite")
MODEL_AVAILABLE = os.path.exists(MODEL_PATH)


def _make_classifier():
    from src.classifier import KSLClassifier
    return KSLClassifier()


def test_buffer_fills():
    """30프레임 미만에서는 예측하지 않아야 한다."""
    if not MODEL_AVAILABLE:
        print("[SKIP] test_buffer_fills — model not found")
        return
    clf = _make_classifier()
    for _ in range(SEQUENCE_LENGTH - 1):
        clf.add_frame(np.zeros(INPUT_DIM, dtype=np.float32))
    result = clf.predict()
    assert result is None, "Buffer not full — should return None"
    print("[PASS] test_buffer_fills")


def test_none_frame():
    """None 랜드마크는 버퍼에 추가되지 않아야 한다."""
    if not MODEL_AVAILABLE:
        print("[SKIP] test_none_frame — model not found")
        return
    clf = _make_classifier()
    clf.add_frame(None)
    assert len(clf.sequence_buffer) == 0
    print("[PASS] test_none_frame")


def test_full_pipeline():
    """30프레임을 채우면 predict()가 오류 없이 실행되어야 한다."""
    if not MODEL_AVAILABLE:
        print("[SKIP] test_full_pipeline — model not found")
        return

    clf = _make_classifier()
    dummy = np.random.rand(INPUT_DIM).astype(np.float32)
    for _ in range(SEQUENCE_LENGTH):
        clf.add_frame(dummy)

    result = clf.predict()
    # 신뢰도 미달 시 None 가능 — 오류 없이 실행되는지만 확인
    print(f"[PASS] test_full_pipeline — result: {result}")


def test_stale_buffer_reset_on_no_hands():
    """연속 NO_HAND_RESET_FRAMES 프레임 양손 미검출 시 버퍼가 비워져야 한다.
    버퍼 로직은 모델 유무와 무관 — 더미 모드에서도 검증 가능."""
    clf = _make_classifier()
    dummy = np.random.rand(INPUT_DIM).astype(np.float32)
    for _ in range(5):
        clf.add_frame(dummy)
    assert len(clf.sequence_buffer) == 5

    for _ in range(NO_HAND_RESET_FRAMES):
        clf.add_frame(None)
    assert len(clf.sequence_buffer) == 0, (
        f"{NO_HAND_RESET_FRAMES}프레임 연속 None 후 버퍼가 비워져야 한다"
    )
    print("[PASS] test_stale_buffer_reset_on_no_hands")


def test_no_hand_counter_resets_on_valid_frame():
    """미검출 카운터는 유효 프레임이 들어오면 리셋 — 임계 미만 끊김은 버퍼 유지."""
    clf = _make_classifier()
    dummy = np.random.rand(INPUT_DIM).astype(np.float32)
    for _ in range(5):
        clf.add_frame(dummy)
    for _ in range(NO_HAND_RESET_FRAMES - 1):
        clf.add_frame(None)
    clf.add_frame(dummy)  # 카운터 리셋
    for _ in range(NO_HAND_RESET_FRAMES - 1):
        clf.add_frame(None)
    assert len(clf.sequence_buffer) == 6, (
        "임계 미만의 끊김으로는 버퍼가 비워지지 않아야 한다"
    )
    print("[PASS] test_no_hand_counter_resets_on_valid_frame")


def test_buffer_cleared_after_recognition():
    """인식 확정 후 시퀀스 버퍼가 초기화되어야 한다."""
    if not MODEL_AVAILABLE:
        print("[SKIP] test_buffer_cleared_after_recognition — model not found")
        return

    clf = _make_classifier()
    dummy = np.random.rand(INPUT_DIM).astype(np.float32)
    for _ in range(SEQUENCE_LENGTH):
        clf.add_frame(dummy)

    clf.predict()
    assert len(clf.sequence_buffer) == 0, "Buffer should be cleared after prediction"
    print("[PASS] test_buffer_cleared_after_recognition")


if __name__ == "__main__":
    test_buffer_fills()
    test_none_frame()
    test_full_pipeline()
    test_stale_buffer_reset_on_no_hands()
    test_no_hand_counter_resets_on_valid_frame()
    test_buffer_cleared_after_recognition()
    print("\nAll tests done.")
