"""
test_hand_tracker_two_hands.py
양손 입력 포맷(131차원) HandTracker 검증.
실 카메라 의존 없이 MediaPipe 결과를 fake 객체로 주입.

mediapipe 설치 환경(.venv) 전용 — CI에서는 실행하지 않는다 (test_smoke.py가
소스 마커·feature_format 검증을 대신 수행).

로컬 실행:
    .venv/bin/python tests/test_hand_tracker_two_hands.py
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    FEATURE_DIM,
    LEFT_SLOT_START,
    RIGHT_SLOT_START,
    WRIST_VEC_START,
    PRESENCE_FLAG_START,
)


# ── Fake MediaPipe 결과 ───────────────────────────────────────

class FakeLandmark:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeHandLandmarks:
    def __init__(self, points):
        # points: list of (x, y, z) tuples, length 21
        self.landmark = [FakeLandmark(*p) for p in points]


class FakeClassification:
    def __init__(self, label, score):
        self.label = label
        self.score = score


class FakeHandedness:
    def __init__(self, label, score):
        self.classification = [FakeClassification(label, score)]


class FakeMediaPipeResult:
    def __init__(self, hand_landmarks_list, handedness_list):
        self.multi_hand_landmarks = hand_landmarks_list or None
        self.multi_handedness = handedness_list or None


def make_hand(wrist_xyz, scale=0.1):
    """손목을 wrist_xyz에 두고 21개 landmark가 손목 주변 scale 범위에 분포된 fake 손."""
    wx, wy, wz = wrist_xyz
    pts = []
    for i in range(21):
        # 손목(0)은 정확히 wrist_xyz, 나머지는 idx에 따라 오프셋
        offset = (i % 7) * (scale / 7)
        pts.append((wx + offset, wy + offset, wz))
    # landmark 9 (중지 MCP)는 손목으로부터 scale 거리에 둔다 — intra-hand scale 기준점
    pts[9] = (wx + scale, wy, wz)
    return FakeHandLandmarks(pts)


_tracker = None

def get_tracker():
    """HandTracker 싱글턴 — MediaPipe Hands 초기화를 1회로 제한."""
    global _tracker
    if _tracker is None:
        from src.hand_tracker import HandTracker
        _tracker = HandTracker()
    return _tracker


def _build(mp_result, mirrored=True, frame_size=(640, 480)):
    """기본 (640,480) — 운용 카메라 해상도로 G6 등방 보정 경로를 검증한다.
    기존 단언은 x축·부호 기반이라 보정 후에도 동일하게 성립한다."""
    return get_tracker()._build_feature_vector(
        mp_result, mirrored=mirrored, frame_size=frame_size
    )


# ── T1: 두 손 happy-path ───────────────────────────────────────

def test_two_hands_returns_131_vector():
    """두 손 모두 검출 시 길이 131, 두 슬롯 비-zero, presence flag 둘 다 1,
    wrist_vec은 scale 평균으로 정규화된 (우손목−좌손목)."""
    # 거울모드(no-swap): MediaPipe 'Left' = 해부학적 좌손 = 화면 작은 x
    left_hand = make_hand((0.3, 0.5, 0.0), scale=0.1)
    right_hand = make_hand((0.7, 0.5, 0.0), scale=0.1)
    mp_result = FakeMediaPipeResult(
        [left_hand, right_hand],
        [FakeHandedness("Left", 0.95), FakeHandedness("Right", 0.95)],
    )

    out = _build(mp_result, mirrored=True)

    assert out is not None
    assert out.shape == (FEATURE_DIM,)
    assert out.dtype == np.float32
    assert np.any(out[LEFT_SLOT_START:RIGHT_SLOT_START] != 0.0)
    assert np.any(out[RIGHT_SLOT_START:WRIST_VEC_START] != 0.0)
    assert out[PRESENCE_FLAG_START] == 1.0
    assert out[PRESENCE_FLAG_START + 1] == 1.0
    # wrist_vec = (0.7-0.3, 0, 0) / mean_scale(0.1) = (4, 0, 0)
    np.testing.assert_allclose(
        out[WRIST_VEC_START:WRIST_VEC_START + 3], [4.0, 0.0, 0.0], atol=1e-5
    )
    print("[PASS] test_two_hands_returns_131_vector")


def test_intra_hand_scale_normalization():
    """landmark 9(중지 MCP)의 손목 상대 거리가 정규화 후 정확히 1이어야 한다."""
    mp_result = FakeMediaPipeResult(
        [make_hand((0.3, 0.5, 0.0), scale=0.25)],  # scale을 바꿔도
        [FakeHandedness("Left", 0.95)],
    )
    out = _build(mp_result)
    lm9 = out[LEFT_SLOT_START + 9 * 3:LEFT_SLOT_START + 9 * 3 + 3]
    np.testing.assert_allclose(np.linalg.norm(lm9), 1.0, atol=1e-5)
    print("[PASS] test_intra_hand_scale_normalization")


# ── T2/T3: 한 손만 검출 ────────────────────────────────────────

def test_left_only():
    mp_result = FakeMediaPipeResult(
        [make_hand((0.3, 0.5, 0.0))],
        [FakeHandedness("Left", 0.95)],
    )
    out = _build(mp_result)
    assert out is not None
    assert np.any(out[LEFT_SLOT_START:RIGHT_SLOT_START] != 0.0)
    assert np.all(out[RIGHT_SLOT_START:WRIST_VEC_START] == 0.0)
    assert np.all(out[WRIST_VEC_START:WRIST_VEC_START + 3] == 0.0)
    assert out[PRESENCE_FLAG_START] == 1.0
    assert out[PRESENCE_FLAG_START + 1] == 0.0
    print("[PASS] test_left_only")


def test_right_only():
    mp_result = FakeMediaPipeResult(
        [make_hand((0.7, 0.5, 0.0))],
        [FakeHandedness("Right", 0.95)],
    )
    out = _build(mp_result)
    assert out is not None
    assert np.all(out[LEFT_SLOT_START:RIGHT_SLOT_START] == 0.0)
    assert np.any(out[RIGHT_SLOT_START:WRIST_VEC_START] != 0.0)
    assert out[PRESENCE_FLAG_START] == 0.0
    assert out[PRESENCE_FLAG_START + 1] == 1.0
    print("[PASS] test_right_only")


# ── T4: 양손 모두 미검출 ───────────────────────────────────────

def test_no_hands_returns_none():
    assert _build(FakeMediaPipeResult([], [])) is None
    assert _build(FakeMediaPipeResult(None, None)) is None
    print("[PASS] test_no_hands_returns_none")


# ── T5: score 미달 → x좌표 fallback ────────────────────────────

def test_low_score_x_fallback():
    """두 손 모두 score 미달이면 손목 x가 작은 손이 좌손 슬롯 (거울모드 기준)."""
    hand_small_x = make_hand((0.2, 0.5, 0.0))
    hand_large_x = make_hand((0.8, 0.6, 0.0))
    mp_result = FakeMediaPipeResult(
        [hand_large_x, hand_small_x],  # 순서 뒤섞어 입력
        [FakeHandedness("Left", 0.4), FakeHandedness("Right", 0.4)],
    )
    out = _build(mp_result)
    assert out is not None
    assert out[PRESENCE_FLAG_START] == 1.0 and out[PRESENCE_FLAG_START + 1] == 1.0
    # 작은 x(0.2)가 LEFT에 배정됐다면 wrist_vec x = (0.8-0.2)/scale > 0
    assert out[WRIST_VEC_START] > 0.0, "x fallback: 작은 x가 좌손 슬롯이어야 한다"
    # y도 검증: 우손(0.6) − 좌손(0.5) > 0
    assert out[WRIST_VEC_START + 1] > 0.0
    print("[PASS] test_low_score_x_fallback")


def test_one_low_score_fills_empty_slot():
    """한 손만 score 미달이면 라벨 확정 손의 반대 슬롯에 배정."""
    confident_left = make_hand((0.3, 0.5, 0.0))
    uncertain = make_hand((0.7, 0.5, 0.0))
    mp_result = FakeMediaPipeResult(
        [confident_left, uncertain],
        [FakeHandedness("Left", 0.95), FakeHandedness("Left", 0.3)],
    )
    out = _build(mp_result)
    assert out[PRESENCE_FLAG_START] == 1.0 and out[PRESENCE_FLAG_START + 1] == 1.0
    assert out[WRIST_VEC_START] > 0.0  # 미달 손(0.7)이 RIGHT로
    print("[PASS] test_one_low_score_fills_empty_slot")


# ── T6: 비거울 입력 시 라벨 swap ───────────────────────────────

def test_unmirrored_label_swap():
    """mirrored=False면 MediaPipe 'Left' 라벨 손이 우손 슬롯에 배정."""
    mp_result = FakeMediaPipeResult(
        [make_hand((0.3, 0.5, 0.0))],
        [FakeHandedness("Left", 0.95)],
    )
    out = _build(mp_result, mirrored=False)
    assert np.all(out[LEFT_SLOT_START:RIGHT_SLOT_START] == 0.0)
    assert np.any(out[RIGHT_SLOT_START:WRIST_VEC_START] != 0.0)
    assert out[PRESENCE_FLAG_START] == 0.0
    assert out[PRESENCE_FLAG_START + 1] == 1.0
    print("[PASS] test_unmirrored_label_swap")


# ── 라벨 충돌 재배정 ───────────────────────────────────────────

def test_handedness_conflict_reassignment():
    """두 손이 모두 'Left'로 분류되면 score 낮은 쪽이 우손 슬롯으로 이동."""
    high_score = make_hand((0.3, 0.5, 0.0))   # score 0.95 → LEFT 유지
    low_score = make_hand((0.7, 0.5, 0.0))    # score 0.80 → RIGHT 재배정
    mp_result = FakeMediaPipeResult(
        [high_score, low_score],
        [FakeHandedness("Left", 0.95), FakeHandedness("Left", 0.80)],
    )
    out = _build(mp_result)
    assert out[PRESENCE_FLAG_START] == 1.0 and out[PRESENCE_FLAG_START + 1] == 1.0
    # 높은 score(x=0.3)가 LEFT, 낮은 score(x=0.7)가 RIGHT → wrist_vec x > 0
    assert out[WRIST_VEC_START] > 0.0
    print("[PASS] test_handedness_conflict_reassignment")


# ── G6: 좌표 비등방성 보정 (PR #9 리뷰 #1) ─────────────────────

def _diagonal_hand(w, h):
    """픽셀 기준 (60, 80) 대각 오프셋(=100px)에 lm9를 둔 fake 손.
    MediaPipe 정규화 좌표(x÷w, y÷h)로 표현 — 등방 보정 검증용."""
    wx, wy = 0.5, 0.5
    pts = [(wx, wy, 0.0)] * 21
    pts[9] = (wx + 60.0 / w, wy + 80.0 / h, 0.0)
    return FakeHandLandmarks(pts)


def test_anisotropy_correction():
    """frame_size=(640,480) 주입 시 (w,h,w) 스케일로 등방 복원 —
    픽셀 (60,80,0)/100px → 정규화 lm9 = (0.6, 0.8, 0)."""
    w, h = 640, 480
    mp_result = FakeMediaPipeResult(
        [_diagonal_hand(w, h)], [FakeHandedness("Left", 0.95)],
    )
    out = get_tracker()._build_feature_vector(
        mp_result, mirrored=True, frame_size=(w, h)
    )
    lm9 = out[LEFT_SLOT_START + 9 * 3:LEFT_SLOT_START + 9 * 3 + 3]
    np.testing.assert_allclose(lm9, [0.6, 0.8, 0.0], atol=1e-5)
    print("[PASS] test_anisotropy_correction")


def test_no_frame_size_skips_correction():
    """frame_size=None이면 무보정 — 입력을 등방 좌표로 간주 (기존 동작)."""
    w, h = 640, 480
    mp_result = FakeMediaPipeResult(
        [_diagonal_hand(w, h)], [FakeHandedness("Left", 0.95)],
    )
    out = get_tracker()._build_feature_vector(
        mp_result, mirrored=True, frame_size=None
    )
    rel = np.array([60.0 / w, 80.0 / h, 0.0], dtype=np.float32)
    expected = rel / np.linalg.norm(rel)  # 비등방 그대로 ≈ (0.490, 0.872, 0)
    lm9 = out[LEFT_SLOT_START + 9 * 3:LEFT_SLOT_START + 9 * 3 + 3]
    np.testing.assert_allclose(lm9, expected, atol=1e-5)
    print("[PASS] test_no_frame_size_skips_correction")


# ── 퇴화 케이스 ────────────────────────────────────────────────

def test_degenerate_hand_dropped():
    """모든 landmark가 한 점에 모인 손(scale≈0)은 미감지로 처리."""
    collapsed = FakeHandLandmarks([(0.5, 0.5, 0.0)] * 21)
    valid = make_hand((0.3, 0.5, 0.0))
    mp_result = FakeMediaPipeResult(
        [collapsed, valid],
        [FakeHandedness("Right", 0.95), FakeHandedness("Left", 0.95)],
    )
    out = _build(mp_result)
    assert out is not None
    assert out[PRESENCE_FLAG_START] == 1.0       # 좌손(valid)만 존재
    assert out[PRESENCE_FLAG_START + 1] == 0.0   # 우손(collapsed)은 탈락

    # 양손 모두 퇴화면 None
    both_collapsed = FakeMediaPipeResult(
        [collapsed], [FakeHandedness("Left", 0.95)],
    )
    assert _build(both_collapsed) is None
    print("[PASS] test_degenerate_hand_dropped")


if __name__ == "__main__":
    test_two_hands_returns_131_vector()
    test_intra_hand_scale_normalization()
    test_left_only()
    test_right_only()
    test_no_hands_returns_none()
    test_low_score_x_fallback()
    test_one_low_score_fills_empty_slot()
    test_unmirrored_label_swap()
    test_handedness_conflict_reassignment()
    test_anisotropy_correction()
    test_no_frame_size_skips_correction()
    test_degenerate_hand_dropped()
    print("\nAll tests done.")
