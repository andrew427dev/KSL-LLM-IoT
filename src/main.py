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

    # 모듈 초기화
    tracker = HandTracker()
    classifier = KSLClassifier()
    builder = SentenceBuilder()
    tts = TTSOutput()
    lcd = LCDDisplay()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

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

            # 6. 디버그 뷰
            tracker.draw_landmarks(frame)
            cv2.putText(frame, f"Buffer: {preview}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("KSL-LLM-IoT", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
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
