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
import threading
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracker import HandTracker
from src.classifier import KSLClassifier
from src.sentence_builder import SentenceBuilder
from src.tts_output import TTSOutput
from src.lcd_display import LCDDisplay
from src.button_input import EVENT_COMPLETE
from config.settings import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, BUZZER_PIN, LED_PIN,
    SENTENCE_PERSONAS, KSL_LABELS_EN, TRIGGER_WORD,
)

# 페르소나별 비프 횟수 — 정중 1회, 친근 2회, 간단 3회 (선언 순서).
# 화면을 보지 않아도 어떤 문체가 적용됐는지 소리로 구분한다.
_PERSONA_BEEPS = {name: i + 1 for i, name in enumerate(SENTENCE_PERSONAS)}

# 마지막으로 출력한 문장 (한국어, 영어) — 완료 버튼 재누름 시 재생용.
_last_sentence = None

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT)  # 부저 동기 시각 피드백 LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[Main] RPi.GPIO not available (non-RPi environment)")
except RuntimeError as e:
    # 핀 점유/권한 등으로 setmode·setup이 실패해도 임포트 크래시 대신 부저 없이 계속.
    GPIO_AVAILABLE = False
    print(f"[Main] GPIO init failed ({e}) — running without buzzer")


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


def beep(times=1, long=False):
    """신호음 + LED 동기 점멸 (비동기). 메인 인식 루프를 블록하지 않는다.

    times: 펄스 횟수 — 페르소나 버튼 피드백(정중1/친근2/간단3)처럼
    화면을 보지 않아도 어떤 입력이 적용됐는지 구분하게 한다.
    long: True면 길게 1회 울린다 — 문장 완료/재생 신호를 단어 인식음과
    청각·시각 모두에서 구분한다.
    LED(LED_PIN)는 부저와 1:1로 켜지고 꺼져, 농인 사용자를 위한 시각 피드백을 준다.
    """
    if not GPIO_AVAILABLE:
        return
    dur = 0.6 if long else 0.05
    threading.Thread(target=_beep_pulse, args=(times, dur), daemon=True).start()


def _beep_pulse(times=1, dur=0.05):
    for i in range(times):
        if i:
            time.sleep(0.1)
        try:
            GPIO.output(BUZZER_PIN, True)
            GPIO.output(LED_PIN, True)   # 부저와 동기 — 시각 피드백
            time.sleep(dur)
            GPIO.output(BUZZER_PIN, False)
            GPIO.output(LED_PIN, False)
        except RuntimeError:
            # 종료 시 GPIO.cleanup() 이후 데몬 스레드가 늦게 도는 경우 — 조용히 중단.
            return


def _handle_control_event(event, builder, lcd, tts):
    """물리 버튼·GUI 키 공통 제어 이벤트 처리.

    event: EVENT_COMPLETE(문장 완료/재생) 또는 페르소나 이름("정중"/"친근"/"간단").
    완료: 버퍼에 단어가 있으면 새 문장 생성, 비어 있으면 마지막 문장을 재생한다
    (완료 버튼 재누름 = 다시 듣기). 둘 다 길게 1회 비프 + LED로 신호한다.
    """
    if event == EVENT_COMPLETE:
        if builder.trigger_sentence():
            beep(long=True)
            lcd.write_line(3, "Generating...")
        elif _last_sentence is not None:
            beep(long=True)
            _output_sentence(_last_sentence, tts, lcd)  # 마지막 문장 재생
        return
    if builder.set_persona(event):
        beep(_PERSONA_BEEPS.get(event, 1))
        print(f"[Main] Sentence persona: {event}")


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

    # 물리 버튼 (RPi): 문장 완료 1 + 페르소나 3. 비-RPi 환경에서는
    # GUI 모드의 SPACE(완료)·p(페르소나 순환) 키가 같은 역할을 한다.
    buttons = None
    if GPIO_AVAILABLE:
        from src.button_input import ButtonInput
        buttons = ButtonInput(GPIO)
        print("[Main] Physical buttons ready "
              "(complete + persona x3, see USER_MANUAL §1.1).")

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
                # 완료 수어는 문장 트리거 — 길게 1회로 단어 인식음과 구분.
                beep(long=True) if word == TRIGGER_WORD else beep()
                # LCD는 한글 렌더링 불가 — 영어 라벨로 표시
                lcd.show_recognition(KSL_LABELS_EN.get(word, "?"), confidence)

                # 3. 단어 버퍼에 추가 (비동기 — 즉시 반환)
                builder.add_word(word)

            # 4. 물리 버튼 처리 — 문장 완료 / 페르소나 전환
            if buttons:
                event = buttons.poll()
                if event:
                    _handle_control_event(event, builder, lcd, tts)

            # 4a. 침묵 트리거 확인 (기본 비활성 — SILENCE_TRIGGER_SEC 참조)
            builder.check_silence_trigger()

            # 4b. 완료된 문장 폴링 (Gemini 워커가 끝났을 때만 반환)
            sentence = builder.poll_sentence()
            if sentence:
                _output_sentence(sentence, tts, lcd)

            # 5. 버퍼 미리보기 표시 (LCD·GUI는 영어 — 한글 렌더링 불가)
            preview = builder.get_buffer_preview(english=True)
            if preview and not result:
                lcd.show_buffer(preview)

            # 6. 디버그 뷰 (헤드리스에선 창 띄우지 않음)
            if not headless:
                tracker.draw_landmarks(frame)
                cv2.putText(frame, f"Buffer: {preview}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("KSL-LLM-IoT", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # SPACE = 문장 완료 (물리 완료 버튼과 동일 경로)
                    _handle_control_event(EVENT_COMPLETE, builder, lcd, tts)
                elif key == ord('p'):
                    # 문장 페르소나 순환 (정중 → 친근 → 간단 → ...)
                    names = list(SENTENCE_PERSONAS)
                    nxt = names[(names.index(builder.persona) + 1) % len(names)]
                    _handle_control_event(nxt, builder, lcd, tts)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        tracker.release()
        try:
            lcd.close()  # LCD 워커 스레드 정상 정지 (정의된 종료 API 사용)
        except Exception:
            pass
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("[Main] System stopped.")


def _output_sentence(sentence, tts, lcd):
    """(한국어, 영어) 문장 출력 — 음성은 한국어, LCD는 영어(한글 렌더링 불가).

    마지막 문장을 보관해 완료 버튼 재누름(버퍼 빈 상태) 시 재생할 수 있게 한다.
    """
    global _last_sentence
    _last_sentence = sentence
    korean, english = sentence
    print(f"\n[Sentence] {korean}  /  {english}\n")
    lcd.show_sentence(english)
    tts.speak(korean)


if __name__ == "__main__":
    main()
