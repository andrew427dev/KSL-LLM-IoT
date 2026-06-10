"""
classifier.py
TFLite 모델로 수화 단어를 실시간 분류합니다.
30프레임 시퀀스를 입력받아 단어 레이블과 신뢰도를 반환합니다.
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
    INPUT_SHAPE, KSL_LABELS, DUPLICATE_FILTER_SEC, NO_HAND_RESET_FRAMES
)


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

        # 30프레임 시퀀스 버퍼
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

        # 양손 미검출 연속 프레임 카운터 — 임계 도달 시 버퍼 리셋 (stale 추론 방지)
        self._no_hand_count = 0

        # 중복 인식 방지
        self._last_word = None
        self._last_word_time = 0.0

    def add_frame(self, landmarks):
        """
        랜드마크 벡터(INPUT_SHAPE[1],)를 시퀀스 버퍼에 추가합니다.
        None(양손 미검출)이 NO_HAND_RESET_FRAMES 연속이면 버퍼를 비웁니다 —
        손이 사라진 뒤 직전 동작의 잔류 프레임으로 추론이 반복되는 것을 막습니다.
        """
        if landmarks is not None:
            self._no_hand_count = 0
            self.sequence_buffer.append(landmarks)
        else:
            self._no_hand_count += 1
            if self._no_hand_count >= NO_HAND_RESET_FRAMES:
                self.sequence_buffer.clear()
                self._no_hand_count = 0

    def predict(self):
        """
        버퍼가 가득 찼을 때 추론을 실행합니다.

        Returns:
            (str, float) — (단어 레이블, 신뢰도) 또는 None
        """
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None

        if self._dummy:
            confidence = 0.99
            label_idx = 0
        else:
            sequence = np.array(self.sequence_buffer, dtype=np.float32)
            sequence = np.expand_dims(sequence, axis=0)  # (1, SEQUENCE_LENGTH, FEATURE_DIM)

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
        now = time.time()
        if label == self._last_word and (now - self._last_word_time) < DUPLICATE_FILTER_SEC:
            return None

        self._last_word = label
        self._last_word_time = now
        self.sequence_buffer.clear()

        return label, confidence
