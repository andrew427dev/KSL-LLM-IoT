"""
train.py
KSL 수화 단어 분류 LSTM 모델 훈련 스크립트.

Usage:
    python model/train.py
"""

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config.settings import KSL_LABELS, SEQUENCE_LENGTH, INPUT_SHAPE

INPUT_DIM = INPUT_SHAPE[1]  # 126 (2 hands × 21 × 3)


def load_dataset(landmarks_dir="data/landmarks", augmented_dir="data/augmented"):
    """
    data/landmarks/ (원본) + data/augmented/ (증강) 에서 CSV를 로드합니다.
    augmented_dir 가 없거나 비어 있으면 원본만 사용합니다.
    각 CSV: 30행(프레임) × 126열 = [LEFT 21×3 | RIGHT 21×3]
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
                    seq = np.loadtxt(path, delimiter=",")
                    if seq.shape == (SEQUENCE_LENGTH, INPUT_DIM):
                        X.append(seq)
                        y.append(label)
                except Exception as e:
                    print(f"[Train] Skip {fname}: {e}")

    # 원본 경고는 landmarks_dir 기준으로만 출력
    for label in KSL_LABELS:
        if not os.path.exists(os.path.join(landmarks_dir, label)):
            print(f"[Train] Warning: No data found for '{label}'")

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(num_classes):
    """LSTM 분류기 모델 구조."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train():
    print("[Train] Loading dataset...")
    X, y_raw = load_dataset()

    if len(X) == 0:
        print("[Train] No data found. Please collect data first.")
        return

    print(f"[Train] Loaded {len(X)} samples across {len(set(y_raw))} classes.")

    # 레이블 인코딩
    le = LabelEncoder()
    le.fit(KSL_LABELS)
    y = le.transform(y_raw)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(KSL_LABELS))

    # Train / Val / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15 / 0.85, random_state=42
    )

    # 모델 빌드
    model = build_model(len(KSL_LABELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
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
        callbacks=callbacks
    )

    # 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n[Train] Test Accuracy: {test_acc:.4f}")

    # TFLite 변환
    print("[Train] Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Float16 양자화
    tflite_model = converter.convert()

    with open("model/ksl_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("[Train] Done. Saved: model/ksl_model.tflite")
    return history


if __name__ == "__main__":
    train()
