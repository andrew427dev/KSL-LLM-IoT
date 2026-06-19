"""
diagnose_live.py
실시간 인식 진단 — 확정 임계값(CONFIDENCE_THRESHOLD)을 무시하고
매 추론의 top-3 (단어, 확률)와 손 검출률을 출력한다.

인식 불량의 원인 분리용:
  - 손 검출률(det)이 낮다 → MediaPipe 단계 문제 (조명·거리·FOV)
  - top1이 정답인데 확률이 낮다 → 분포 차이 (임계값/데이터 보강 검토)
  - top3에 정답이 없다 → 모델 일반화 실패 또는 동작 차이

DISPLAY가 있으면(VNC 등) 미리보기 창에 랜드마크와 top-3을 표시한다.

Usage (RPi):
    DISPLAY=:0 .venv/bin/python tools/diagnose_live.py [지속시간초=180]
"""
import sys
import time
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.hand_tracker import HandTracker
from src.classifier import KSLClassifier
from config.settings import KSL_LABELS, KSL_LABELS_EN, CAMERA_INDEX


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0

    clf = KSLClassifier()  # 시간 리샘플 경로 — 운용과 동일한 전처리
    assert not clf._dummy, "운용 모델(ksl_model.tflite)이 필요하다"
    inp = clf.input_details[0]
    out = clf.output_details[0]

    cap = cv2.VideoCapture(CAMERA_INDEX)
    tracker = HandTracker()
    frames = detected = 0
    t0 = time.time()
    show = bool(os.environ.get("DISPLAY"))
    overlay = ""

    print(f"[Diag] 시작 — {duration:.0f}초 동안 top-3 출력 "
          f"(시간 리샘플 적용, 임계값 무시)", flush=True)
    while time.time() - t0 < duration:
        ok, frame = cap.read()
        if not ok:
            print("[Diag] 카메라 read 실패", flush=True)
            break
        frame = cv2.flip(frame, 1)
        lm = tracker.extract_landmarks(frame)
        frames += 1
        if lm is not None:
            detected += 1
        clf.add_frame(lm)

        # 0.5초마다 한 번씩 추론 결과 출력 (1초 창이 준비됐을 때)
        if lm is not None and frames % 5 == 0:
            seq = clf._resample_window(time.monotonic())
            if seq is None:
                continue
            sequence = np.expand_dims(seq, axis=0)
            clf.interpreter.set_tensor(inp["index"], sequence)
            clf.interpreter.invoke()
            p = clf.interpreter.get_tensor(out["index"])[0]
            top = p.argsort()[-3:][::-1]
            elapsed = time.time() - t0
            print(
                f"[{elapsed:5.1f}s] det {detected}/{frames} | "
                + "  ".join(f"{KSL_LABELS[i]}={p[i]:.2f}" for i in top),
                flush=True,
            )
            overlay = "  ".join(
                f"{KSL_LABELS_EN.get(KSL_LABELS[i], '?')}:{p[i]:.2f}" for i in top
            )

        if show:
            tracker.draw_landmarks(frame)
            cv2.putText(frame, f"hand: {'O' if lm is not None else '-'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if lm is not None else (0, 0, 255), 2)
            if overlay:
                cv2.putText(frame, overlay, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("KSL Diagnose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    fps = frames / max(time.time() - t0, 1e-6)
    print(f"[Diag] 종료 — 손 검출 {detected}/{frames} "
          f"({detected / max(frames, 1):.0%}), 평균 {fps:.1f} FPS", flush=True)
    cap.release()
    tracker.release()


if __name__ == "__main__":
    main()
