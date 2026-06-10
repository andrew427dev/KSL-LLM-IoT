"""
test_classifier.py
KSLClassifier 유닛 테스트 — 시간 기반 리샘플링 포함.

분류기는 (timestamp, vector) 버퍼에서 최근 WINDOW_SEC(1.0초)을
SEQUENCE_LENGTH 포인트로 보간한다. 가상 시계를 주입해 저FPS(9fps)
환경을 시뮬레이션한다. 더미 모드로 동작하므로 모델 파일 불필요.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SEQUENCE_LENGTH, INPUT_SHAPE, NO_HAND_RESET_FRAMES,
    PRESENCE_FLAG_START, RIGHT_SLOT_START, WRIST_VEC_START,
)
from src.classifier import KSLClassifier, WINDOW_SEC, MIN_RESAMPLE_FRAMES

INPUT_DIM = INPUT_SHAPE[1]  # FEATURE_DIM = 131


def _make_classifier():
    return KSLClassifier()


def _vec(value=0.5, left=True, right=True):
    """presence 불변식을 지키는 테스트용 131벡터."""
    v = np.zeros(INPUT_DIM, dtype=np.float32)
    if left:
        v[:RIGHT_SLOT_START] = value
        v[PRESENCE_FLAG_START] = 1.0
    if right:
        v[RIGHT_SLOT_START:WRIST_VEC_START] = value
        v[PRESENCE_FLAG_START + 1] = 1.0
    if left and right:
        v[WRIST_VEC_START:WRIST_VEC_START + 3] = value
    return v


def _fill_window(clf, fps=9.0, duration=None, start=100.0, value=0.5):
    """가상 fps로 duration초 동안 프레임 적재. 마지막 시각을 반환."""
    duration = WINDOW_SEC if duration is None else duration
    n = max(int(fps * duration), 1)
    t = start
    for i in range(n):
        t = start + i / fps
        clf.add_frame(_vec(value), now=t)
    return t


# ── 시간 창 게이팅 ─────────────────────────────────────────────

def test_no_predict_before_window_filled():
    """창(1초)을 채우기 전에는 예측하지 않아야 한다."""
    clf = _make_classifier()
    _fill_window(clf, fps=9.0, duration=WINDOW_SEC * 0.4)  # 0.4초만
    assert clf.predict(now=100.0 + WINDOW_SEC * 0.4) is None
    print("[PASS] test_no_predict_before_window_filled")


def test_predict_after_window_filled():
    """저FPS(9fps)로도 1초 창이 차면 예측이 실행돼야 한다 (더미 모드)."""
    clf = _make_classifier()
    t_end = _fill_window(clf, fps=9.0, duration=WINDOW_SEC * 1.1)
    result = clf.predict(now=t_end)
    assert result is not None, "1초 창 + 9fps에서 예측이 나와야 한다"
    print(f"[PASS] test_predict_after_window_filled — {result}")


def test_none_frame():
    """None 랜드마크는 버퍼에 추가되지 않아야 한다."""
    clf = _make_classifier()
    clf.add_frame(None)
    assert len(clf.sequence_buffer) == 0
    print("[PASS] test_none_frame")


# ── 리샘플링 수치 검증 (모델 불필요 — _resample_window 직접) ──

def test_resample_shape_and_interpolation():
    """9fps 입력 → (SEQUENCE_LENGTH, 131) 보간, 값이 두 프레임 사이를 보간."""
    clf = _make_classifier()
    # 0.0~1.1초 동안 값이 0→1로 선형 증가하는 프레임 (10fps)
    fps, start = 10.0, 50.0
    n = int(fps * 1.1) + 1
    for i in range(n):
        t = start + i / fps
        clf.add_frame(_vec(value=i / (n - 1)), now=t)
    now = start + 1.1
    seq = clf._resample_window(now)
    assert seq is not None and seq.shape == (SEQUENCE_LENGTH, INPUT_DIM)
    # 보간된 손 슬롯 값이 단조 증가(시간을 따라가는 보간)인지
    col = seq[:, 0]
    assert np.all(np.diff(col) >= -1e-6), "보간 값은 시간순으로 단조여야 한다"
    print("[PASS] test_resample_shape_and_interpolation")


def test_resample_presence_invariants():
    """보간 후 presence flag는 0/1이고, 부재 슬롯·wrist_vec은 0이어야 한다."""
    clf = _make_classifier()
    fps, start = 10.0, 50.0
    # 앞 절반 양손, 뒤 절반 왼손만 → 경계에서 flag 보간 발생
    for i in range(12):
        t = start + i / fps
        clf.add_frame(_vec(0.7, left=True, right=(i < 6)), now=t)
    seq = clf._resample_window(start + 1.1)
    assert seq is not None
    flags = seq[:, PRESENCE_FLAG_START:PRESENCE_FLAG_START + 2]
    assert np.all((flags == 0.0) | (flags == 1.0)), "flag는 0/1 이산값"
    right_absent = flags[:, 1] == 0.0
    assert np.all(seq[right_absent, RIGHT_SLOT_START:WRIST_VEC_START] == 0.0)
    assert np.all(seq[right_absent, WRIST_VEC_START:WRIST_VEC_START + 3] == 0.0)
    print("[PASS] test_resample_presence_invariants")


def test_resample_rejects_stale_window():
    """마지막 프레임이 오래됐으면(손 사라진 뒤) 추론하지 않아야 한다."""
    clf = _make_classifier()
    t_end = _fill_window(clf, fps=10.0, duration=1.1)
    assert clf._resample_window(t_end) is not None
    assert clf._resample_window(t_end + WINDOW_SEC) is None, "stale 창은 거부"
    print("[PASS] test_resample_rejects_stale_window")


# ── 버퍼 리셋 정책 ─────────────────────────────────────────────

def test_stale_buffer_reset_on_no_hands():
    """연속 NO_HAND_RESET_FRAMES 프레임 양손 미검출 시 버퍼가 비워져야 한다."""
    clf = _make_classifier()
    _fill_window(clf, fps=9.0, duration=0.5)
    assert len(clf.sequence_buffer) > 0
    for _ in range(NO_HAND_RESET_FRAMES):
        clf.add_frame(None)
    assert len(clf.sequence_buffer) == 0
    print("[PASS] test_stale_buffer_reset_on_no_hands")


def test_no_hand_counter_resets_on_valid_frame():
    """미검출 카운터는 유효 프레임이 들어오면 리셋 — 임계 미만 끊김은 버퍼 유지."""
    clf = _make_classifier()
    _fill_window(clf, fps=9.0, duration=0.5, start=10.0)
    n0 = len(clf.sequence_buffer)
    for _ in range(NO_HAND_RESET_FRAMES - 1):
        clf.add_frame(None)
    clf.add_frame(_vec(), now=11.0)  # 카운터 리셋
    for _ in range(NO_HAND_RESET_FRAMES - 1):
        clf.add_frame(None)
    assert len(clf.sequence_buffer) == n0 + 1
    print("[PASS] test_no_hand_counter_resets_on_valid_frame")


def test_buffer_cleared_after_recognition():
    """인식 확정 후 시퀀스 버퍼가 초기화되어야 한다 (더미 모드)."""
    clf = _make_classifier()
    t_end = _fill_window(clf, fps=9.0, duration=1.1)
    result = clf.predict(now=t_end)
    assert result is not None
    assert len(clf.sequence_buffer) == 0
    print("[PASS] test_buffer_cleared_after_recognition")


if __name__ == "__main__":
    test_no_predict_before_window_filled()
    test_predict_after_window_filled()
    test_none_frame()
    test_resample_shape_and_interpolation()
    test_resample_presence_invariants()
    test_resample_rejects_stale_window()
    test_stale_buffer_reset_on_no_hands()
    test_no_hand_counter_resets_on_valid_frame()
    test_buffer_cleared_after_recognition()
    print("\nAll tests done.")
