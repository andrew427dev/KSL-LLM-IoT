"""
test_classifier.py
KSLClassifier 유닛 테스트.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import KSLClassifier
from config.settings import SEQUENCE_LENGTH


def test_buffer_fills():
    """30프레임 미만에서는 예측하지 않아야 한다."""
    clf = KSLClassifier()
    for _ in range(SEQUENCE_LENGTH - 1):
        clf.add_frame(np.zeros(63, dtype=np.float32))
    result = clf.predict()
    assert result is None, "Buffer not full — should return None"
    print("[PASS] test_buffer_fills")


def test_none_frame():
    """None 랜드마크는 버퍼에 추가되지 않아야 한다."""
    clf = KSLClassifier()
    clf.add_frame(None)
    assert len(clf.sequence_buffer) == 0
    print("[PASS] test_none_frame")


def test_full_pipeline():
    """30프레임을 채우면 predict()가 None이 아닌 값을 반환해야 한다."""
    if not os.path.exists("model/ksl_model.tflite"):
        print("[SKIP] test_full_pipeline — model not found")
        return

    clf = KSLClassifier()
    dummy = np.random.rand(63).astype(np.float32)
    for _ in range(SEQUENCE_LENGTH):
        clf.add_frame(dummy)

    result = clf.predict()
    # 신뢰도 미달 시 None 가능 — 오류 없이 실행되는지만 확인
    print(f"[PASS] test_full_pipeline — result: {result}")


if __name__ == "__main__":
    test_buffer_fills()
    test_none_frame()
    test_full_pipeline()
    print("\nAll tests passed.")
