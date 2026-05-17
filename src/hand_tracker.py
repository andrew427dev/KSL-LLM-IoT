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

견고성 처리:
- handedness score < HANDEDNESS_SCORE_THRESHOLD 인 손은 라벨이 불확실하므로
  좌/우 swap 노이즈를 방지하기 위해 폐기.
- 두 손이 동일 라벨로 분류된 충돌 케이스(드물지만 발생) 시 score 높은 쪽을
  해당 라벨에 두고, 점수 낮은 쪽은 반대편 슬롯이 비어 있으면 거기로 재배정.
  재배정 시 stderr에 warning을 한 줄 출력한다.
"""

import sys
import cv2
import mediapipe as mp
import numpy as np
from config.settings import NUM_HANDS, NUM_LANDMARKS, NUM_AXES

_PER_HAND_DIM = NUM_LANDMARKS * NUM_AXES  # 63

# handedness 신뢰도 임계값 — 미달 시 해당 손은 미감지로 처리.
# Week 2 수집 중 raw 분포를 보고 0.6~0.8 범위에서 재조정 가능.
HANDEDNESS_SCORE_THRESHOLD = 0.7


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
            None — 양손 모두 감지되지 않거나 모두 score 미달인 경우.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_result = self.hands.process(rgb)

        if not self._last_result.multi_hand_landmarks:
            return None

        # 후보 수집: score 임계값 통과한 손만 (label, score, normalized_63) 보관.
        # MediaPipe 내부 버그로 multi_handedness 길이가 multi_hand_landmarks와
        # 불일치할 경우 zip이 짧은 쪽으로 잘려 자연스럽게 dropped 됨 (방어적).
        candidates = []
        for hand_landmarks, handedness in zip(
            self._last_result.multi_hand_landmarks,
            self._last_result.multi_handedness,
        ):
            c = handedness.classification[0]
            if c.score < HANDEDNESS_SCORE_THRESHOLD:
                continue
            candidates.append((c.label, c.score, self._normalize_hand(hand_landmarks)))

        if not candidates:
            return None

        # 라벨 충돌 처리: 같은 라벨이 2건 이상이면 비어 있는 반대편으로 옮긴다.
        # score 가장 낮은 쪽을 옮겨, 가장 신뢰도 높은 손이 본래 라벨에 유지되도록 한다.
        by_label = {"Left": [], "Right": []}
        for label, score, arr in candidates:
            by_label[label].append((score, arr))

        for src, dst in (("Left", "Right"), ("Right", "Left")):
            while len(by_label[src]) > 1 and not by_label[dst]:
                by_label[src].sort(key=lambda x: x[0])  # 점수 오름차순
                moved_score, moved_arr = by_label[src].pop(0)
                by_label[dst].append((moved_score, moved_arr))
                print(
                    f"[HandTracker] handedness conflict — both hands labeled '{src}', "
                    f"reassigning lower-score hand (score={moved_score:.2f}) to '{dst}'.",
                    file=sys.stderr,
                )

        # 슬롯별 최고 점수 손만 사용 (충돌 후 잔여 중복은 폐기).
        left = (max(by_label["Left"], key=lambda x: x[0])[1]
                if by_label["Left"] else np.zeros(_PER_HAND_DIM, dtype=np.float32))
        right = (max(by_label["Right"], key=lambda x: x[0])[1]
                 if by_label["Right"] else np.zeros(_PER_HAND_DIM, dtype=np.float32))

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
