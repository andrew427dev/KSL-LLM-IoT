"""
evaluate.py
훈련된 TFLite 모델의 정확도, FPS, 혼동 행렬을 평가합니다.

Usage:
    python model/evaluate.py
"""

import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from model.train import load_dataset
from config.settings import KSL_LABELS, MODEL_PATH, SEQUENCE_LENGTH


def evaluate():
    print("[Evaluate] Loading model and dataset...")
    X, y_raw = load_dataset()

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    latencies = []

    for seq in X:
        inp = np.expand_dims(seq, axis=0).astype(np.float32)
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append(time.time() - start)
        y_pred.append(KSL_LABELS[np.argmax(output)])

    # 정확도 리포트
    print("\n" + classification_report(y_raw, y_pred, target_names=KSL_LABELS))
    print(f"Avg Inference Latency: {np.mean(latencies)*1000:.2f} ms")
    print(f"Estimated FPS: {1.0/np.mean(latencies):.1f}")

    # 혼동 행렬 저장
    cm = confusion_matrix(y_raw, y_pred, labels=KSL_LABELS)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(KSL_LABELS)))
    ax.set_yticks(range(len(KSL_LABELS)))
    ax.set_xticklabels(KSL_LABELS, rotation=90, fontsize=8)
    ax.set_yticklabels(KSL_LABELS, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — KSL Classifier')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png", dpi=150)
    print("[Evaluate] Saved: model/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
