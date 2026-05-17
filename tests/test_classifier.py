"""
test_classifier.py
KSLClassifier 유닛 테스트.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SEQUENCE_LENGTH, INPUT_SHAPE

INPUT_DIM = INPUT_SHAPE[1]  # 126 (2 hands × 21 × 3)

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
    test_buffer_cleared_after_recognition()
    print("\nAll tests done.")
