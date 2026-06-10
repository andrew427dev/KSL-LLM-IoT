"""
train.py
KSL 수화 단어 분류 LSTM 모델 훈련 스크립트.

Usage:
    python model/train.py

레이블 인코딩 주의:
    추론(src/classifier.py)이 `KSL_LABELS[label_idx]`로 역매핑하므로
    학습 레이블 인덱스는 반드시 KSL_LABELS의 원본 순서를 따라야 한다.
    sklearn LabelEncoder는 유니코드 정렬을 수행해 인덱스가 어긋난다 — 사용 금지.
"""

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config.settings import KSL_LABELS, SEQUENCE_LENGTH, FEATURE_DIM, INPUT_SHAPE

RANDOM_SEED = 42

# KSL_LABELS 원본 순서 기반 인코딩 — classifier.py의 KSL_LABELS[label_idx]와 일치
LABEL_TO_IDX = {label: i for i, label in enumerate(KSL_LABELS)}


def encode_labels(y_raw):
    """라벨 문자열 목록 → KSL_LABELS 원본 순서 기준 정수 인덱스."""
    return np.array([LABEL_TO_IDX[label] for label in y_raw], dtype=np.int32)


def load_dataset(landmarks_dir="data/landmarks", augmented_dir="data/augmented"):
    """
    data/landmarks/ (원본) + data/augmented/ (증강) 에서 CSV를 로드합니다.
    augmented_dir 가 없거나 비어 있으면 원본만 사용합니다.
    각 CSV: SEQUENCE_LENGTH행 × FEATURE_DIM열 (src/feature_format.py 레이아웃)
    """
    X, y = [], []

    dirs_to_load = [landmarks_dir]
    if augmented_dir and os.path.exists(augmented_dir):
        dirs_to_load.append(augmented_dir)

    for src_dir in dirs_to_load:
        for label in KSL_LABELS:
            label_dir = os.path.join(src_dir, label)
            if not os.path.exists(label_dir):
                continue

            for fname in os.listdir(label_dir):
                if not fname.endswith(".csv"):
                    continue
                path = os.path.join(label_dir, fname)
                try:
                    # pandas C 파서 — np.loadtxt 대비 대용량 로딩 수 배 빠름
                    seq = pd.read_csv(path, header=None, dtype=np.float32).values
                    if seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                        X.append(seq)
                        y.append(label)
                except Exception as e:
                    print(f"[Train] Skip {fname}: {e}")

    # 원본 경고는 landmarks_dir 기준으로만 출력
    for label in KSL_LABELS:
        if not os.path.exists(os.path.join(landmarks_dir, label)):
            print(f"[Train] Warning: No data found for '{label}'")

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(num_classes, batch_size=None):
    """LSTM 분류기 모델 구조.

    batch_size: 학습 시 None(dynamic). TFLite 변환 시 1로 고정 —
    dynamic batch에서는 LSTM 내부 TensorListReserve의 element_shape가
    static이 아니어서 변환이 실패한다 (TF 2.15 확인).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE, batch_size=batch_size),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def convert_and_verify_tflite(model, output_path="model/ksl_model.tflite"):
    """Keras 모델 → TFLite 변환 후, 로드·shape 검증을 통과해야 저장한다.

    변환은 batch=1 고정 복제 모델로 수행한다 (build_model docstring 참조).
    런타임(src/classifier.py)도 (1, SEQUENCE_LENGTH, FEATURE_DIM) 단건 추론이라
    배치 고정에 따른 기능 손실은 없다.
    """
    fixed = build_model(len(KSL_LABELS), batch_size=1)
    fixed.set_weights(model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(fixed)
    # float16 양자화는 적용하지 않는다 — TF 2.15에서 LSTM + float16 조합의
    # 변환 패스가 메모리 폭주로 OOM kill됨 (24GB 컨테이너에서 실측).
    # 본 모델은 float32로도 ~740KB라 양자화 이득이 없다.
    tflite_model = converter.convert()

    # 저장 전 검증: 인터프리터 로드 + 입출력 shape 일치
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_shape = tuple(interpreter.get_input_details()[0]["shape"])
    out_shape = tuple(interpreter.get_output_details()[0]["shape"])
    expected_in = (1, SEQUENCE_LENGTH, FEATURE_DIM)
    expected_out = (1, len(KSL_LABELS))
    assert in_shape == expected_in, f"TFLite input {in_shape} != {expected_in}"
    assert out_shape == expected_out, f"TFLite output {out_shape} != {expected_out}"

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[Train] TFLite verified ({in_shape} -> {out_shape}). Saved: {output_path}")


def train():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("[Train] Loading dataset...")
    X, y_raw = load_dataset()

    if len(X) == 0:
        print("[Train] No data found. Please collect data first.")
        return

    print(f"[Train] Loaded {len(X)} samples across {len(set(y_raw))} classes.")

    y = encode_labels(y_raw)

    # Train / Val / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15 / 0.85,
        random_state=RANDOM_SEED, stratify=y_train
    )

    # 클래스 불균형 보정 — AI Hub 단어별 시연자 수 차이 대비
    counts = np.bincount(y_train, minlength=len(KSL_LABELS)).astype(np.float64)
    present = counts > 0
    class_weight = {
        i: (counts[present].mean() / counts[i]) if counts[i] > 0 else 0.0
        for i in range(len(KSL_LABELS))
    }

    # 모델 빌드
    model = build_model(len(KSL_LABELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 콜백
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            "model/best_model.keras", save_best_only=True, monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ]

    # 훈련
    print("[Train] Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks
    )

    # 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n[Train] Test Accuracy: {test_acc:.4f}")

    # TFLite 변환 + 검증
    print("[Train] Converting to TFLite...")
    convert_and_verify_tflite(model)
    return history


if __name__ == "__main__":
    train()
