"""
main.py
KSL-LLM-IoT 메인 실행 파일.
전체 파이프라인을 초기화하고 실시간 인식 루프를 실행합니다.

Usage:
    python src/main.py
"""

import cv2
import time
import sys
import os
import shutil
import subprocess
import select
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracker import HandTracker
from src.classifier import KSLClassifier
from src.sentence_builder import SentenceBuilder
from src.tts_output import TTSOutput
from src.lcd_display import LCDDisplay
from config.settings import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, BUZZER_PIN
)

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[Main] RPi.GPIO not available (non-RPi environment)")


class CameraReader:
    """USB 웹캠(cv2) / rpicam-vid 파이프 / picamera2 순서로 자동 선택."""

    def __init__(self, index=0, width=640, height=480, fps=30):
        self.width, self.height = width, height
        self.mode = None
        self._cv = None
        self._picam = None
        self._rpicam = None
        self._frame_bytes = width * height * 3 // 2   # YUV420 = w*h*1.5
        self._first_chunk = None

        # 1차: OpenCV (USB 웹캠)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                self._cv = cap
                self.mode = "opencv"
                print(f"[Camera] Using OpenCV VideoCapture(index={index})")
                return
            cap.release()

        # 2차: rpicam-vid 서브프로세스 (venv에 libcamera-python이 없는 환경 대응)
        if shutil.which("rpicam-vid") is not None:
            try:
                cmd = [
                    "rpicam-vid",
                    "--timeout", "0",
                    "--width", str(width),
                    "--height", str(height),
                    "--framerate", str(fps),
                    "--codec", "yuv420",
                    "--nopreview",
                    "-o", "-",
                ]
                self._rpicam = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                # 백엔드 검증: 첫 프레임을 5초 안에 받아야 함
                data = self._read_exact(self._frame_bytes, timeout=5.0)
                if data is None:
                    raise RuntimeError("rpicam-vid produced no frame within 5s")
                self._first_chunk = data
                self.mode = "rpicam"
                print("[Camera] Using rpicam-vid subprocess (Pi CSI YUV420)")
                return
            except Exception as e:
                print(f"[Camera] rpicam-vid backend skipped: {e}")
                if self._rpicam is not None:
                    self._rpicam.terminate()
                    self._rpicam = None

        # 3차: picamera2 (Python 바인딩이 갖춰진 환경)
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()

            # Camera Module 3 등 AF 지원 카메라: Continuous AF
            try:
                from libcamera import controls
                picam.set_controls({
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfRange": controls.AfRangeEnum.Normal,
                    "AfSpeed": controls.AfSpeedEnum.Fast,
                })
                print("[Camera] Autofocus: Continuous (Module 3+ detected)")
            except Exception:
                pass  # AF 미지원 카메라(v1/v2)

            self._picam = picam
            self.mode = "picamera2"
            print("[Camera] Using picamera2 (Pi CSI)")
            return
        except Exception as e:
            raise RuntimeError(f"No camera available (all backends failed): {e}")

    def _read_exact(self, n, timeout=None):
        """rpicam-vid stdout에서 정확히 n바이트를 읽음. timeout 초 안에 못 채우면 None."""
        fd = self._rpicam.stdout.fileno()
        buf = bytearray()
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        while len(buf) < n:
            remaining = (deadline - time.monotonic()) if deadline else None
            if remaining is not None and remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def read(self):
        if self.mode == "opencv":
            return self._cv.read()
        if self.mode == "rpicam":
            if self._first_chunk is not None:
                data, self._first_chunk = self._first_chunk, None
            else:
                data = self._read_exact(self._frame_bytes, timeout=2.0)
                if data is None:
                    return False, None
            yuv = np.frombuffer(data, dtype=np.uint8).reshape(
                (self.height * 3 // 2, self.width)
            )
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            return True, bgr
        if self.mode == "picamera2":
            frame = self._picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        return False, None

    def release(self):
        if self._cv is not None:
            self._cv.release()
        if self._rpicam is not None:
            self._rpicam.terminate()
            try:
                self._rpicam.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._rpicam.kill()
        if self._picam is not None:
            self._picam.stop()


def beep():
    """수화 인식 확정 신호음."""
    if GPIO_AVAILABLE:
        GPIO.output(BUZZER_PIN, True)
        time.sleep(0.1)
        GPIO.output(BUZZER_PIN, False)


def main():
    print("=" * 40)
    print("  KSL-LLM-IoT System Starting...")
    print("=" * 40)

    # SSH/PuTTY 등 X 디스플레이가 없으면 자동으로 헤드리스 모드.
    # 강제 토글: KSL_HEADLESS=1 / KSL_HEADLESS=0
    _force = os.environ.get("KSL_HEADLESS")
    if _force is not None:
        headless = _force == "1"
    else:
        headless = not os.environ.get("DISPLAY")
    if headless:
        print("[Main] Headless mode (no GUI window). Ctrl+C to quit.")

    # 모듈 초기화
    tracker = HandTracker()
    classifier = KSLClassifier()
    builder = SentenceBuilder()
    tts = TTSOutput()
    lcd = LCDDisplay()

    cap = CameraReader(index=CAMERA_INDEX,
                       width=CAMERA_WIDTH,
                       height=CAMERA_HEIGHT,
                       fps=CAMERA_FPS)

    lcd.write_line(0, "KSL-LLM-IoT Ready")
    lcd.write_line(1, "Show your sign...")
    lcd.write_line(2, "")
    lcd.write_line(3, "")

    print("[Main] System ready. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드

            # 1. 랜드마크 추출
            landmarks = tracker.extract_landmarks(frame)
            classifier.add_frame(landmarks)

            # 2. 분류 (30프레임 누적 시)
            result = classifier.predict()
            if result:
                word, confidence = result
                print(f"[Classifier] {word} ({confidence:.1%})")
                beep()
                lcd.show_recognition(word, confidence)

                # 3. 단어 버퍼에 추가
                sentence = builder.add_word(word)
                if sentence:
                    _output_sentence(sentence, tts, lcd)

            # 4. 침묵 트리거 확인
            sentence = builder.check_silence_trigger()
            if sentence:
                _output_sentence(sentence, tts, lcd)

            # 5. 버퍼 미리보기 표시
            preview = builder.get_buffer_preview()
            if preview and not result:
                lcd.show_buffer(preview)

            # 6. 디버그 뷰 (헤드리스에선 창 띄우지 않음)
            if not headless:
                tracker.draw_landmarks(frame)
                cv2.putText(frame, f"Buffer: {preview}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("KSL-LLM-IoT", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        tracker.release()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("[Main] System stopped.")


def _output_sentence(sentence, tts, lcd):
    print(f"\n[Sentence] {sentence}\n")
    lcd.show_sentence(sentence)
    tts.speak(sentence)


if __name__ == "__main__":
    main()
