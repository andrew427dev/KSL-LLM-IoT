"""
classifier.py
TFLite 모델로 수화 단어를 실시간 분류합니다.

시간 정규화: 학습 데이터(AI Hub)는 30fps × 30프레임 = 1.0초 창이지만,
런타임 FPS는 환경에 따라 다르다 (RPi 4B + MediaPipe 실측 ~8-9 FPS).
프레임 개수로 버퍼를 채우면 저FPS 환경에서 1초 동작이 4초 창으로
늘어져 학습 분포와 어긋난다 — 그래서 버퍼를 (timestamp, vector)로 들고,
추론 시 최근 1.0초 구간을 SEQUENCE_LENGTH 포인트로 선형 보간해
학습과 동일한 시간 창을 재구성한다.
"""

import os
import sys
import numpy as np
import time
from collections import deque

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from config.settings import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, SEQUENCE_LENGTH,
    KSL_LABELS, DUPLICATE_FILTER_SEC, NO_HAND_RESET_FRAMES,
    TRAIN_FPS, LEFT_SLOT_START, RIGHT_SLOT_START,
    WRIST_VEC_START, PRESENCE_FLAG_START,
)

# 인식 시간 창(초) — 학습 데이터의 시퀀스 길이와 동일한 물리적 시간
WINDOW_SEC = SEQUENCE_LENGTH / TRAIN_FPS  # 1.0초

# 보간에 필요한 최소 실측 프레임 수 — 이보다 적으면 동작 정보가 부족
MIN_RESAMPLE_FRAMES = 8

_PER_HAND_DIM = RIGHT_SLOT_START - LEFT_SLOT_START  # 63


class KSLClassifier:
    def __init__(self):
        self._dummy = not os.path.exists(MODEL_PATH)
        if self._dummy:
            print(f"[Classifier] No model at '{MODEL_PATH}'. "
                  f"DUMMY MODE: always predicts '{KSL_LABELS[0]}' (smoke test).")
            self.interpreter = None
            self.input_details = None
            self.output_details = None
        else:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        # (timestamp, 131-vector) 버퍼 — 창 길이의 2배 시간을 넉넉히 보관
        self.sequence_buffer = deque(maxlen=int(TRAIN_FPS * WINDOW_SEC * 2) + 10)

        # 양손 미검출 연속 프레임 카운터 — 임계 도달 시 버퍼 리셋 (stale 추론 방지)
        self._no_hand_count = 0

        # 중복 인식 방지
        self._last_word = None
        self._last_word_time = 0.0

    def add_frame(self, landmarks, now=None):
        """
        랜드마크 벡터(FEATURE_DIM,)를 타임스탬프와 함께 버퍼에 추가합니다.
        None(양손 미검출)이 NO_HAND_RESET_FRAMES 연속이면 버퍼를 비웁니다.

        now: 테스트용 가상 시계 주입 (기본 time.monotonic()).
        """
        if landmarks is not None:
            self._no_hand_count = 0
            t = time.monotonic() if now is None else now
            self.sequence_buffer.append((t, landmarks))
        else:
            self._no_hand_count += 1
            if self._no_hand_count >= NO_HAND_RESET_FRAMES:
                self.sequence_buffer.clear()
                self._no_hand_count = 0

    def _resample_window(self, now):
        """최근 WINDOW_SEC 구간을 SEQUENCE_LENGTH 포인트로 선형 보간.

        Returns: (SEQUENCE_LENGTH, FEATURE_DIM) float32 또는 None(데이터 부족).
        """
        if len(self.sequence_buffer) < MIN_RESAMPLE_FRAMES:
            return None
        ts = np.array([t for t, _ in self.sequence_buffer])
        # 창을 충분히 채웠는지 — 동작 초반의 성급한 추론 방지
        if now - ts[0] < WINDOW_SEC * 0.9:
            return None
        # 창 밖(너무 오래된) 프레임만 남았으면 stale — 추론하지 않음
        if now - ts[-1] > WINDOW_SEC * 0.5:
            return None

        X = np.stack([v for _, v in self.sequence_buffer]).astype(np.float32)
        target = np.linspace(now - WINDOW_SEC, now, SEQUENCE_LENGTH)

        idx = np.searchsorted(ts, target, side="right")
        idx = np.clip(idx, 1, len(ts) - 1)
        t0, t1 = ts[idx - 1], ts[idx]
        w = np.clip(
            (target - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0
        ).astype(np.float32)[:, None]
        seq = X[idx - 1] * (1.0 - w) + X[idx] * w

        # presence flag 재이산화 + 부재 슬롯·wrist_vec 불변식 복원
        # (model/augment.py:time_warp와 동일 규칙)
        flags = np.round(seq[:, PRESENCE_FLAG_START:PRESENCE_FLAG_START + 2])
        seq[:, PRESENCE_FLAG_START:PRESENCE_FLAG_START + 2] = flags
        left_absent = flags[:, 0] == 0.0
        right_absent = flags[:, 1] == 0.0
        seq[left_absent, LEFT_SLOT_START:LEFT_SLOT_START + _PER_HAND_DIM] = 0.0
        seq[right_absent, RIGHT_SLOT_START:RIGHT_SLOT_START + _PER_HAND_DIM] = 0.0
        seq[left_absent | right_absent, WRIST_VEC_START:WRIST_VEC_START + 3] = 0.0
        return seq

    def predict(self, now=None):
        """
        최근 1초 창이 준비됐을 때 추론을 실행합니다.

        Returns:
            (str, float) — (단어 레이블, 신뢰도) 또는 None
        """
        now = time.monotonic() if now is None else now

        if self._dummy:
            # 더미 모드: 시간 창과 무관하게 버퍼만 차면 고정 예측 (smoke test)
            if len(self.sequence_buffer) < MIN_RESAMPLE_FRAMES:
                return None
            confidence = 0.99
            label_idx = 0
        else:
            seq = self._resample_window(now)
            if seq is None:
                return None
            sequence = np.expand_dims(seq, axis=0)  # (1, SEQUENCE_LENGTH, FEATURE_DIM)

            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], sequence)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            except (ValueError, RuntimeError) as e:
                # shape 불일치·모델 손상 등 — 전체 파이프라인 정지 대신 해당 추론만 폐기
                print(f"[Classifier] TFLite inference failed: {e}", file=sys.stderr)
                self.sequence_buffer.clear()
                return None

            confidence = float(np.max(output))
            label_idx = int(np.argmax(output))

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        label = KSL_LABELS[label_idx]

        # 중복 필터: 동일 단어가 N초 이내에 다시 인식되면 무시
        wall = time.time()
        if label == self._last_word and (wall - self._last_word_time) < DUPLICATE_FILTER_SEC:
            return None

        self._last_word = label
        self._last_word_time = wall
        self.sequence_buffer.clear()

        return label, confidence
