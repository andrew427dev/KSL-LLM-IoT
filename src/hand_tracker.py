"""
hand_tracker.py
MediaPipe Hands로 양손(좌·우)을 동시에 추출한다.

출력: 한 프레임당 131차원 벡터 (src/feature_format.py 레이아웃 참조)
    [LEFT_63 | RIGHT_63 | wrist_vec_3 | presence_2]
MediaPipe 정규화 좌표는 비등방(x÷너비, y÷높이)이므로 (w,h,w) 스케일로
등방 복원 후 조립한다 (HANDOFF §1.6 G6 — 학습 데이터와 기하 정합).
각 손은 손목(landmark 0) 상대좌표를 intra-hand scale(‖landmark9−landmark0‖)로
나눠 정규화한다. 미감지 손은 zero + presence=0. 양손 모두 미감지면 None.

MediaPipe의 handedness 라벨은 *입력 영상이 거울 모드(selfie)임을 가정*하고
부여된다. 본 프로젝트의 main.py / collect_data.py는 모두 cv2.flip(frame, 1)
이후 extract_landmarks()를 호출하므로(mirrored=True), MediaPipe의 'Left' =
사용자의 해부학적 왼손에 일치한다 — swap 불필요 (PR #7 실기 검증).
mirrored=False로 호출하면(비거울 입력) 라벨을 swap한다.

견고성 처리:
- handedness score < HANDEDNESS_SCORE_THRESHOLD 인 손은 라벨을 신뢰하지 않고
  x좌표 fallback으로 슬롯을 배정한다 (거울모드에서 작은 x = 사용자 좌손).
- 두 손이 동일 라벨로 분류된 충돌 케이스(드물지만 발생) 시 score 높은 쪽을
  해당 라벨에 두고, 점수 낮은 쪽은 반대편 슬롯이 비어 있으면 거기로 재배정.
  재배정 시 stderr에 warning을 한 줄 출력한다.
"""

import sys
import cv2
import mediapipe as mp
import numpy as np

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    NUM_HANDS, HANDEDNESS_SCORE_THRESHOLD, MP_MODEL_COMPLEXITY,
)
from src.feature_format import build_feature_vector


class HandTracker:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=NUM_HANDS,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._last_result = None  # extract_landmarks의 최신 결과 캐시

    def extract_landmarks(self, frame, mirrored=True):
        """
        프레임에서 양손 랜드마크를 추출해 131차원 feature 벡터로 반환한다.

        Args:
            frame: BGR 프레임. 호출자가 거울모드(cv2.flip)를 이미 적용했으면
                   mirrored=True(기본). 비거울 입력이면 mirrored=False.
        Returns:
            np.ndarray of shape (FEATURE_DIM,) 또는 None(양손 모두 미감지).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_result = self.hands.process(rgb)
        h, w = frame.shape[:2]
        return self._build_feature_vector(
            self._last_result, mirrored=mirrored, frame_size=(w, h)
        )

    def _build_feature_vector(self, result, mirrored=True, frame_size=None):
        """MediaPipe 결과 객체 → 131차원 벡터 (테스트에서 fake 객체 주입 가능).

        frame_size=(w, h): MediaPipe 정규화 좌표는 비등방(x÷너비, y÷높이) —
        640×480에서 y가 x 대비 4/3배 스케일이라 학습 데이터(AI Hub 등방
        미터 좌표)와 기하가 어긋난다. (w, h, w) 스케일로 픽셀 단위 등방
        좌표로 복원한다 (MediaPipe z는 x와 동일 스케일 — w 적용).
        None이면 입력을 이미 등방인 좌표로 간주해 보정하지 않는다.
        (PR #9 리뷰 #1, HANDOFF §1.6 G6)
        """
        if not result.multi_hand_landmarks:
            return None

        axis_scale = None
        if frame_size is not None:
            fw, fh = frame_size
            axis_scale = np.array([fw, fh, fw], dtype=np.float32)

        handednesses = result.multi_handedness or []

        # 후보 수집: (label_or_None, score, coords(21,3)).
        # score 미달·handedness 누락 손은 label=None → x좌표 fallback 대상.
        labeled = []
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            if axis_scale is not None:
                coords = coords * axis_scale
            label, score = None, 0.0
            if i < len(handednesses):
                c = handednesses[i].classification[0]
                score = c.score
                if c.score >= HANDEDNESS_SCORE_THRESHOLD:
                    label = c.label
                    if not mirrored:
                        # 비거울 입력 — MediaPipe의 selfie 가정과 어긋나므로 swap
                        label = "Right" if label == "Left" else "Left"
            labeled.append((label, score, coords))

        by_label = {"Left": [], "Right": []}
        unlabeled = []
        for label, score, coords in labeled:
            if label is None:
                unlabeled.append((score, coords))
            else:
                by_label[label].append((score, coords))

        # 라벨 충돌 처리: 같은 라벨이 2건 이상이면 score 낮은 쪽을 빈 반대편으로.
        for src, dst in (("Left", "Right"), ("Right", "Left")):
            while len(by_label[src]) > 1 and not by_label[dst]:
                by_label[src].sort(key=lambda x: x[0])  # 점수 오름차순
                moved_score, moved_coords = by_label[src].pop(0)
                by_label[dst].append((moved_score, moved_coords))
                print(
                    f"[HandTracker] handedness conflict — both hands labeled '{src}', "
                    f"reassigning lower-score hand (score={moved_score:.2f}) to '{dst}'.",
                    file=sys.stderr,
                )

        # 슬롯별 최고 점수 손만 사용 (충돌 후 잔여 중복은 폐기).
        left = max(by_label["Left"], key=lambda x: x[0])[1] if by_label["Left"] else None
        right = max(by_label["Right"], key=lambda x: x[0])[1] if by_label["Right"] else None

        # x좌표 fallback: 라벨 불확실 손을 빈 슬롯에 배정.
        # 거울모드 기준 화면 작은 x = 사용자 좌손 (비거울이면 좌표 자체가 반대라
        # 동일 비교가 성립하지 않으나, 현 파이프라인은 항상 mirrored=True).
        if unlabeled:
            unlabeled.sort(key=lambda sc: sc[1][0, 0])  # 손목 x 오름차순
            for _score, coords in unlabeled:
                if left is None:
                    left = coords
                elif right is None:
                    right = coords
                # 두 슬롯 모두 차 있으면 폐기

        return build_feature_vector(left, right)

    def draw_landmarks(self, frame):
        """extract_landmarks()의 캐시 결과로 양손 랜드마크를 시각화 (이중 처리 없음)."""
        if self._last_result and self._last_result.multi_hand_landmarks:
            for hand in self._last_result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def release(self):
        self.hands.close()
