"""
test_label_encoding.py
학습 레이블 인코딩이 KSL_LABELS 원본 순서를 따르는지 회귀 검증.

배경: sklearn LabelEncoder는 유니코드 정렬을 수행해 학습 인덱스가
추론(src/classifier.py의 `KSL_LABELS[label_idx]`)과 어긋나는 치명적
버그가 있었다. 본 테스트는 그 재발을 막는다.

tensorflow 설치 환경 전용 (model/train.py가 TF를 모듈 레벨에서 import).
CI에서는 test_smoke.py의 소스 마커 검사("LabelEncoder" 부재)가 대신 수행.

로컬 실행:
    .venv/bin/python tests/test_label_encoding.py
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import KSL_LABELS

try:
    from model.train import encode_labels, LABEL_TO_IDX
    TRAIN_IMPORTABLE = True
except ImportError as e:
    TRAIN_IMPORTABLE = False
    _IMPORT_ERROR = e


def test_encoding_follows_original_order():
    """encode_labels는 KSL_LABELS의 원본 순서(정렬 아님)를 따라야 한다."""
    if not TRAIN_IMPORTABLE:
        print(f"[SKIP] test_encoding_follows_original_order — {_IMPORT_ERROR}")
        return
    encoded = encode_labels(KSL_LABELS)
    np.testing.assert_array_equal(encoded, np.arange(len(KSL_LABELS)))
    print("[PASS] test_encoding_follows_original_order")


def test_roundtrip_with_classifier_decoding():
    """학습 인코딩 → 추론 역매핑(KSL_LABELS[idx])이 항등이어야 한다."""
    if not TRAIN_IMPORTABLE:
        print(f"[SKIP] test_roundtrip_with_classifier_decoding — {_IMPORT_ERROR}")
        return
    for word in KSL_LABELS:
        idx = LABEL_TO_IDX[word]
        assert KSL_LABELS[idx] == word, (
            f"round-trip 불일치: '{word}' → idx {idx} → '{KSL_LABELS[idx]}'"
        )
    print("[PASS] test_roundtrip_with_classifier_decoding")


def test_original_order_differs_from_sorted():
    """전제 검증: KSL_LABELS가 정렬 순서와 다름 — 같다면 본 테스트가 무력화된다."""
    assert list(KSL_LABELS) != sorted(KSL_LABELS), (
        "KSL_LABELS가 우연히 정렬 순서와 일치 — LabelEncoder 회귀를 탐지할 수 없음"
    )
    print("[PASS] test_original_order_differs_from_sorted")


if __name__ == "__main__":
    test_original_order_differs_from_sorted()
    test_encoding_follows_original_order()
    test_roundtrip_with_classifier_decoding()
    print("\nAll tests done.")
