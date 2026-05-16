"""
hand_tracker.py
MediaPipe Hands로 양손(좌·우)을 동시에 추출한다.

출력 레이아웃: 한 프레임에 대해 [LEFT_63 | RIGHT_63] = 126차원 벡터.
각 손은 자신의 손목(landmark 0) 기준으로 상대 좌표 정규화한다.
미감지 손은 zero-pad. 양손 모두 미감지면 None.

MediaPipe의 handedness 라벨은 *입력 영상이 거울 모드(selfie)임을 가정*하고
부여된다. 본 프로젝트의 main.py / collect_data.py는 모두 cv2.flip(frame, 1)
이후 extract_landmarks()를 호출하므로, MediaPipe의 'Left' = 사용자의
해부학적 왼손에 일치한다.
"""

import cv2
import mediapipe as mp
import numpy as np
from config.settings import NUM_HANDS, NUM_LANDMARKS, NUM_AXES

_PER_HAND_DIM = NUM_LANDMARKS * NUM_AXES  # 63


class HandTracker:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=NUM_HANDS,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._last_result = None  # extract_landmarks의 최신 결과 캐시

    def extract_landmarks(self, frame):
        """
        프레임에서 양손 랜드마크를 추출한다.

        Returns:
            np.ndarray of shape (126,) — [LEFT_63 | RIGHT_63] 순서.
            None — 양손 모두 감지되지 않은 경우.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_result = self.hands.process(rgb)

        if not self._last_result.multi_hand_landmarks:
            return None

        left = np.zeros(_PER_HAND_DIM, dtype=np.float32)
        right = np.zeros(_PER_HAND_DIM, dtype=np.float32)
        found = False

        for hand_landmarks, handedness in zip(
            self._last_result.multi_hand_landmarks,
            self._last_result.multi_handedness,
        ):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            arr = self._normalize_hand(hand_landmarks)
            if label == "Left":
                left = arr
            else:
                right = arr
            found = True

        if not found:
            return None

        return np.concatenate([left, right])

    @staticmethod
    def _normalize_hand(hand_landmarks):
        """손목(landmark 0) 기준 상대 좌표로 정규화한 21×3 = 63차원 벡터."""
        wrist = hand_landmarks.landmark[0]
        flat = []
        for lm in hand_landmarks.landmark:
            flat.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(flat, dtype=np.float32)

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
