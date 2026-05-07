"""
hand_tracker.py
MediaPipe Hands를 사용해 실시간으로 손 랜드마크를 추출합니다.
"""

import cv2
import mediapipe as mp
import numpy as np
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self._last_result = None  # extract_landmarks의 최신 결과 캐시

    def extract_landmarks(self, frame):
        """
        프레임에서 손 랜드마크 21개 추출.

        Returns:
            np.ndarray of shape (63,) — 정규화된 x, y, z 좌표 플랫 배열
            None — 손이 감지되지 않은 경우
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_result = self.hands.process(rgb)

        if not self._last_result.multi_hand_landmarks:
            return None

        hand = self._last_result.multi_hand_landmarks[0]
        landmarks = []

        # 손목(0번) 기준 상대 좌표 정규화
        wrist = hand.landmark[0]
        for lm in hand.landmark:
            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        return np.array(landmarks, dtype=np.float32)

    def draw_landmarks(self, frame):
        """extract_landmarks()의 캐시 결과로 랜드마크를 시각화 (이중 처리 없음)."""
        if self._last_result and self._last_result.multi_hand_landmarks:
            for hand in self._last_result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def release(self):
        self.hands.close()
