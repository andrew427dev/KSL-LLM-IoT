"""
make_dummy_model.py
파이프라인 smoke test용 더미 TFLite 모델 생성.
학습 없이 정상 입출력 shape의 .tflite를 만들어 main.py 통합 동작을 검증합니다.
가중치는 항상 KSL_LABELS[0](첫 라벨)을 ~100% 신뢰도로 출력하도록 조작됩니다.

Usage:
    python make_dummy_model.py
"""

import os
import sys
import tempfile
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.settings import INPUT_SHAPE, KSL_LABELS

NUM_CLASSES = len(KSL_LABELS)
FLAT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1]

inputs = tf.keras.Input(shape=INPUT_SHAPE)
x = tf.keras.layers.Flatten()(inputs)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# 항상 첫 클래스(KSL_LABELS[0])를 출력하도록 bias 조작
weights = np.zeros((FLAT_SIZE, NUM_CLASSES), dtype=np.float32)
bias = np.zeros(NUM_CLASSES, dtype=np.float32)
bias[0] = 10.0  # softmax(10 vs 0,...) ≈ [0.9999, ~0, ...]
model.layers[-1].set_weights([weights, bias])

# 검증: 임의 입력에 대해 출력 확인
test_input = np.random.rand(1, *INPUT_SHAPE).astype(np.float32)
out = model(test_input).numpy()[0]
print(f"[Verify] argmax={int(np.argmax(out))} (expect 0), "
      f"confidence={float(np.max(out)):.4f} (expect ~1.0)")
print(f"[Verify] predicted label: {KSL_LABELS[int(np.argmax(out))]}")

# Keras 3 호환: from_keras_model 우회, SavedModel 중간 단계 경유
with tempfile.TemporaryDirectory() as tmp:
    model.export(tmp)
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
    tflite_model = converter.convert()

os.makedirs("model", exist_ok=True)
out_path = "model/ksl_model.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_model)

print(f"\n[Done] Saved: {out_path} ({len(tflite_model):,} bytes)")
print(f"       This is a DUMMY model for pipeline smoke testing only.")
print(f"       It always predicts '{KSL_LABELS[0]}' regardless of input.")
