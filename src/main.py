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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracker import HandTracker
from src.classifier import KSLClassifier
from src.sentence_builder import SentenceBuilder
from src.tts_output import TTSOutput
from src.lcd_display import LCDDisplay
from src.button_input import EVENT_COMPLETE, EVENT_UNDO
from config.settings import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, BUZZER_PIN, LED_PIN,
    SENTENCE_PERSONAS, KSL_LABELS_EN, TRIGGER_WORD,
)

# 페르소나별 비프 횟수 — 정중 1회, 친근 2회 (선언 순서). 초기화(undo)는 3회로 구분.
# 화면을 보지 않아도 어떤 문체가 적용됐는지 소리로 구분한다.
_PERSONA_BEEPS = {name: i + 1 for i, name in enumerate(SENTENCE_PERSONAS)}
# 페르소나 한국어 → LCD 표기(영어). HD44780은 한글 렌더링 불가.
_PERSONA_EN = {"정중": "Polite", "친근": "Friendly"}

# 브라우저 라이브 프리뷰(MJPEG)용 최신 주석 프레임 홀더 (KSL_STREAM 활성 시 사용).
_stream = {"frame": None, "lock": threading.Lock()}

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

    times: 펄스 횟수 — 페르소나 버튼 피드백(정중1/친근2)·초기화(3)처럼
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


def _start_stream_server(port=8080):
    """주석 프레임(_stream["frame"])을 MJPEG로 서빙 — 브라우저 라이브 프리뷰.

    카메라는 메인 루프가 단독 점유하므로, 별도 프리뷰 대신 인식 화면 그대로를
    네트워크로 내보낸다. 헤드리스에서도 동작(브라우저로 확인).
    """
    holder = _stream

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path != "/":
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with holder["lock"]:
                        f = holder["frame"]
                    if f is None:
                        time.sleep(0.05); continue
                    ok, jpg = cv2.imencode(".jpg", f,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.wfile.write(b"--frame\r\n"
                                         b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _handle_control_event(event, builder, lcd, tts):
    """물리 버튼·GUI 키 공통 제어 이벤트 처리.

    event: EVENT_COMPLETE(문장 완료/재생), EVENT_UNDO(마지막 단어 제거),
    또는 페르소나 이름("정중"/"친근").
    완료: 버퍼에 단어가 있으면 새 문장 생성, 비어 있으면 마지막 단어를 현재
    페르소나로 다시 생성한다(완료 재누름 = 바뀐 문체로 재출력). 둘 다 길게 1회 비프+LED.
    초기화: 버퍼의 마지막 단어 1개를 제거(오인식 복구), 짧게 3회 비프 + LED.
    """
    if event == EVENT_COMPLETE:
        # 버퍼에 단어가 있으면 새 생성, 없으면 마지막 단어를 현재 페르소나로 재생성.
        if builder.trigger_sentence() or builder.regenerate_last():
            beep(long=True)
            lcd.write_line(3, "Generating...")
            return True  # 호출자가 인식 파이프라인을 리셋하도록 신호
        return False
    if event == EVENT_UNDO:
        removed = builder.undo_last_word()
        if removed:
            beep(3)  # 초기화: 짧게 3회 (단어1·완료길게·페르소나1/2와 구분)
            print(f"[Main] Undo: removed '{removed}'")
            lcd.write_line(3, f"Undo -{KSL_LABELS_EN.get(removed, '?')}")
        return
    if builder.set_persona(event):
        beep(_PERSONA_BEEPS.get(event, 1))
        lcd.show_persona(_PERSONA_EN.get(event, event))  # LCD 상단에 현재 문체 표시
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

    # 물리 버튼 (RPi): 완료 1 + 초기화 1 + 페르소나 2. 비-RPi 환경에서는
    # GUI 모드의 SPACE(완료)·z(초기화)·p(페르소나 순환) 키가 같은 역할을 한다.
    buttons = None
    if GPIO_AVAILABLE:
        from src.button_input import ButtonInput
        buttons = ButtonInput(GPIO)
        print("[Main] Physical buttons ready "
              "(complete + undo + persona x2, see USER_MANUAL §1.1).")

    # 브라우저 라이브 프리뷰 (MJPEG) — KSL_STREAM=1(기본 8080) 또는 포트 번호.
    # 인식 화면(카메라+랜드마크+버퍼)을 그대로 스트리밍한다(헤드리스에서도 동작).
    stream_port = os.environ.get("KSL_STREAM")
    stream_enabled = bool(stream_port)
    if stream_enabled:
        try:
            port = 8080 if stream_port in ("1", "") else int(stream_port)
        except ValueError:
            port = 8080
        _start_stream_server(port)
        print(f"[Main] MJPEG live preview: http://0.0.0.0:{port}/")

    cap = CameraReader(index=CAMERA_INDEX,
                       width=CAMERA_WIDTH,
                       height=CAMERA_HEIGHT,
                       fps=CAMERA_FPS)

    lcd.write_line(0, "KSL-LLM-IoT Ready")
    lcd.write_line(1, "Show your sign...")
    lcd.write_line(2, "")
    lcd.write_line(3, "")
    lcd.show_persona(_PERSONA_EN.get(builder.persona, builder.persona))  # 시작 문체 표시

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
                if word == TRIGGER_WORD:
                    # 완료 수어로 생성 시작 — 인식 리셋(잔여 손동작 누적 방지)
                    classifier.reset_recognition()

            # 4. 물리 버튼 처리 — 문장 완료 / 페르소나 전환
            if buttons:
                event = buttons.poll()
                if event:
                    if _handle_control_event(event, builder, lcd, tts):
                        classifier.reset_recognition()  # 완료/재생성 직후 인식 리셋

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

            # 6. 화면 표시(GUI) / 브라우저 스트림용 주석 프레임 생성
            #    GUI 모드이거나 스트림이 켜져 있으면 랜드마크·버퍼를 그린다.
            if (not headless) or stream_enabled:
                tracker.draw_landmarks(frame)
                cv2.putText(frame, f"Buffer: {preview}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if stream_enabled:
                with _stream["lock"]:
                    _stream["frame"] = frame.copy()

            if not headless:
                cv2.imshow("KSL-LLM-IoT", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # SPACE = 문장 완료 (물리 완료 버튼과 동일 경로)
                    if _handle_control_event(EVENT_COMPLETE, builder, lcd, tts):
                        classifier.reset_recognition()
                elif key == ord('p'):
                    # 문장 페르소나 순환 (정중 → 친근 → ...)
                    names = list(SENTENCE_PERSONAS)
                    nxt = names[(names.index(builder.persona) + 1) % len(names)]
                    _handle_control_event(nxt, builder, lcd, tts)
                elif key == ord('z'):
                    # z = 초기화(마지막 단어 제거, 물리 초기화 버튼과 동일 경로)
                    _handle_control_event(EVENT_UNDO, builder, lcd, tts)

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
    """(한국어, 영어) 문장 출력 — 음성은 한국어, LCD는 영어(한글 렌더링 불가)."""
    korean, english = sentence
    print(f"\n[Sentence] {korean}  /  {english}\n")
    lcd.show_sentence(english)
    tts.speak(korean)


if __name__ == "__main__":
    main()
