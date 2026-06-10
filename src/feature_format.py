"""
feature_format.py
131차원 입력 feature 조립 — 추론(hand_tracker)·학습 데이터 변환(convert_aihub)이
공유하는 단일 소스. 여기 로직이 바뀌면 양쪽이 동시에 바뀌므로 train/serve
전처리 불일치(skew)가 구조적으로 차단된다.

레이아웃 (config/settings.py 상수와 일치):
    [0..62]    좌손 21랜드마크 × (x,y,z) — 손목 상대 / intra-hand scale 정규화
    [63..125]  우손 동일
    [126..128] 우손목 − 좌손목 벡터 — 두 손 scale 평균으로 정규화
    [129]      좌손 presence flag (0/1)
    [130]      우손 presence flag (0/1)

intra-hand scale = ‖landmark[9](중지 MCP) − landmark[0](손목)‖₂.
손 크기·카메라 거리·좌표계 단위(MediaPipe 0~1 정규화 vs AI Hub 미터)에
invariant한 표현을 만든다.

numpy 외 의존성 없음 — mediapipe가 없는 학습 서버에서도 import 가능.
"""

import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    NUM_LANDMARKS,
    FEATURE_DIM,
    LEFT_SLOT_START,
    RIGHT_SLOT_START,
    WRIST_VEC_START,
    PRESENCE_FLAG_START,
)

_PER_HAND_DIM = RIGHT_SLOT_START - LEFT_SLOT_START  # 63

# scale이 이보다 작으면 퇴화 케이스(랜드마크 붕괴) — 해당 손을 미감지로 처리
DEGENERATE_SCALE = 1e-6


def normalize_hand(coords):
    """
    한 손의 절대좌표를 손목 상대 + intra-hand scale로 정규화한다.

    Args:
        coords: (21, 3) array-like — 절대좌표 (단위 무관).
    Returns:
        (normalized_63, scale) — 정규화된 (63,) float32 벡터와 scale 값.
        퇴화 케이스(scale < DEGENERATE_SCALE)면 (None, None).
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (NUM_LANDMARKS, 3):
        return None, None

    rel = coords - coords[0]
    scale = float(np.linalg.norm(rel[9]))
    if scale < DEGENERATE_SCALE:
        return None, None

    return (rel / scale).flatten().astype(np.float32), scale


def build_feature_vector(left_coords, right_coords):
    """
    좌/우 손 절대좌표에서 131차원 feature 벡터를 조립한다.

    Args:
        left_coords:  (21, 3) array-like 또는 None (좌손 미감지).
        right_coords: (21, 3) array-like 또는 None (우손 미감지).
    Returns:
        (FEATURE_DIM,) float32 벡터.
        양손 모두 미감지·퇴화면 None — 호출자는 해당 프레임을 시퀀스에서 제외한다.
    """
    left_norm = left_scale = None
    right_norm = right_scale = None

    if left_coords is not None:
        left_norm, left_scale = normalize_hand(left_coords)
    if right_coords is not None:
        right_norm, right_scale = normalize_hand(right_coords)

    if left_norm is None and right_norm is None:
        return None

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    if left_norm is not None:
        vec[LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND_DIM] = left_norm
        vec[PRESENCE_FLAG_START] = 1.0
    if right_norm is not None:
        vec[RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND_DIM] = right_norm
        vec[PRESENCE_FLAG_START + 1] = 1.0

    if left_norm is not None and right_norm is not None:
        left_wrist = np.asarray(left_coords, dtype=np.float32)[0]
        right_wrist = np.asarray(right_coords, dtype=np.float32)[0]
        mean_scale = (left_scale + right_scale) / 2.0
        vec[WRIST_VEC_START:WRIST_VEC_START + 3] = (
            (right_wrist - left_wrist) / mean_scale
        )

    return vec
