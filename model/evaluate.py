"""
evaluate.py
훈련된 TFLite 모델의 정확도, FPS, 혼동 행렬을 평가합니다.

평가 집합: model/holdout.json(train.py가 기록한 held-out 시연자 원본)이
있으면 그 집합만 평가한다 — train에 노출되지 않은 시연자라 일반화 지표다.
manifest가 없으면 전수 평가로 폴백하되, 학습 데이터가 포함되어 수치가
부풀려지므로 판정에 사용하지 않는다 (PR #9 리뷰 #6).

Usage:
    python model/evaluate.py
"""

import json
import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from model.train import load_dataset, HOLDOUT_MANIFEST
from config.settings import KSL_LABELS, MODEL_PATH, SEQUENCE_LENGTH, FEATURE_DIM


def _load_holdout(manifest_path):
    """train.py가 기록한 held-out 시연자 원본 CSV만 로드한다."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    X, y = [], []
    skipped = 0
    for path in manifest.get("test_files", []):
        label = os.path.basename(os.path.dirname(path))
        try:
            seq = pd.read_csv(path, header=None, dtype=np.float32).values
        except Exception:
            skipped += 1
            continue
        if seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
            X.append(seq)
            y.append(label)
        else:
            skipped += 1
    if skipped:
        print(f"[Evaluate] Warning: manifest 파일 {skipped}개 로드 실패 — "
              f"데이터 디렉터리가 학습 시점과 다르다 (재변환 후 재학습 필요).")
    return np.array(X, dtype=np.float32), np.array(y), manifest


def evaluate():
    print("[Evaluate] Loading model and dataset...")
    if os.path.exists(HOLDOUT_MANIFEST):
        X, y_raw, manifest = _load_holdout(HOLDOUT_MANIFEST)
        print(f"[Evaluate] Held-out 시연자 {manifest['holdout_groups']} "
              f"원본 {len(X)}개 평가 — train 비노출 일반화 지표.")
    else:
        X, y_raw, _files = load_dataset()
        print("[Evaluate] Warning: holdout manifest 없음 — 학습 데이터 포함 전수 평가. "
              "수치가 부풀려지므로 판정에 사용 금지.")

    if len(X) == 0:
        print("[Evaluate] 평가할 샘플이 없다 — 중단.")
        return

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    latencies = []

    for seq in X:
        inp = np.expand_dims(seq, axis=0).astype(np.float32)
        start = time.time()
        # 각 시퀀스를 독립 분류 — fused LSTM 상태가 샘플 간 이월되면
        # 라벨 정렬된 평가셋에서 같은 라벨 잔류상태가 정확도를 부풀린다
        # (상태 누수 0.94 vs 정상 0.72). 추론 코드(classifier.py)와 동일 정책.
        if hasattr(interpreter, "reset_all_variables"):
            interpreter.reset_all_variables()
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append(time.time() - start)
        y_pred.append(KSL_LABELS[np.argmax(output)])

    # 정확도 리포트
    # labels= 명시 — sklearn은 미지정 시 라벨을 정렬해 target_names와 어긋난다
    print("\n" + classification_report(
        y_raw, y_pred, labels=KSL_LABELS, target_names=KSL_LABELS, zero_division=0
    ))
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
