"""
collect_data.py
KSL 수화 데이터 수집 스크립트.
카메라 앞에서 각 수화 단어를 수행하면 랜드마크를 CSV로 저장합니다.

Usage:
    python collect_data.py --word 안녕 --samples 100
"""

import cv2
import numpy as np
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.hand_tracker import HandTracker
from config.settings import (
    KSL_LABELS, SEQUENCE_LENGTH, CAMERA_INDEX, PRESENCE_FLAG_START,
)


def collect(word, num_samples, output_dir="data/landmarks"):
    if word not in KSL_LABELS:
        print(f"[Collect] '{word}' is not in KSL_LABELS. Check config/settings.py")
        return

    save_dir = os.path.join(output_dir, word)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".csv")])
    print(f"[Collect] Word: '{word}' | Existing: {existing} | Target: {num_samples}")

    tracker = HandTracker()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    sample_count = existing
    sequence_buffer = []
    collecting = False

    print("\n  [SPACE] — 수집 시작/정지")
    print("  [q]     — 종료\n")

    while sample_count < existing + num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        landmarks = tracker.extract_landmarks(frame)

        status = f"Collecting: {collecting} | Saved: {sample_count - existing}/{num_samples}"
        cv2.putText(frame, f"Word: {word}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 좌/우 presence flag 라이브 표시 — 어느 손이 잡히는지 즉시 확인
        if landmarks is not None:
            lp = landmarks[PRESENCE_FLAG_START] > 0.5
            rp = landmarks[PRESENCE_FLAG_START + 1] > 0.5
            hands_status = f"Hands  L:{'O' if lp else '-'}  R:{'O' if rp else '-'}"
        else:
            hands_status = "Hands  L:-  R:-"
        cv2.putText(frame, hands_status, (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        if collecting and landmarks is not None:
            sequence_buffer.append(landmarks)
            cv2.putText(frame, f"Frame: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                seq = np.array(sequence_buffer)
                fname = os.path.join(save_dir, f"{sample_count:04d}.csv")
                np.savetxt(fname, seq, delimiter=",")
                print(f"  Saved: {fname}")
                sample_count += 1
                sequence_buffer = []
                collecting = False

        tracker.draw_landmarks(frame)
        cv2.imshow(f"KSL Data Collector — {word}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            collecting = not collecting
            sequence_buffer = []
            print(f"  Collecting: {collecting}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print(f"\n[Collect] Done. Total saved for '{word}': {sample_count - existing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", required=True, help="수화 단어 (KSL_LABELS 중 하나)")
    parser.add_argument("--samples", type=int, default=100, help="수집할 샘플 수")
    args = parser.parse_args()
    collect(args.word, args.samples)
