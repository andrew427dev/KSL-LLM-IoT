"""
lcd_display.py
I2C LCD 20x4 디스플레이 제어 모듈.
인식된 단어 버퍼와 생성된 문장을 LCD에 출력한다.

모든 공개 출력 메서드는 워커 스레드 큐로 작업을 넘긴다.
I2C 한 화면 갱신은 약 100ms 블로킹이 발생하므로, 메인 인식 루프가
이를 직접 감수하지 않도록 격리한다. 큐가 가득 차면 가장 오래된
항목을 버려 최신 화면 상태만 반영한다.
"""

import smbus2
import time
import queue
import threading
from config.settings import LCD_I2C_ADDRESS, LCD_NUM_COLS, LCD_NUM_ROWS

# LCD 명령 상수
LCD_CHR = 1  # 데이터 모드
LCD_CMD = 0  # 명령 모드
LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100

LCD_LINE_ADDR = [0x80, 0xC0, 0x94, 0xD4]  # 20x4 행 주소


class LCDDisplay:
    def __init__(self):
        try:
            self.bus = smbus2.SMBus(1)
            self._init_lcd()
            self.available = True
        except Exception as e:
            print(f"[LCD] Init failed (running without LCD): {e}")
            self.available = False

        # 비동기 I2C 쓰기 큐
        self._queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── 공개 API: 모두 큐에 작업 넣고 즉시 반환 ────────────────────
    def clear(self):
        self._enqueue(self._clear_sync)

    def write_line(self, line_num, text):
        self._enqueue(lambda: self._write_line_sync(line_num, text))

    def show_recognition(self, word, confidence):
        self._enqueue(lambda: self._show_recognition_sync(word, confidence))

    def show_buffer(self, buffer_preview):
        self._enqueue(lambda: self._show_buffer_sync(buffer_preview))

    def show_sentence(self, sentence):
        self._enqueue(lambda: self._show_sentence_sync(sentence))

    def close(self):
        self._stop.set()

    # ── 워커 / 큐 처리 ─────────────────────────────────────────────
    def _enqueue(self, fn):
        try:
            self._queue.put_nowait(fn)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(fn)
            except queue.Full:
                pass

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                fn = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                fn()
            except Exception as e:
                print(f"[LCD] worker error: {e}")

    # ── 동기(워커 내부 실행용) ─────────────────────────────────────
    def _clear_sync(self):
        if not self.available:
            return
        self._write_byte(0x01, LCD_CMD)
        time.sleep(0.05)

    def _write_line_sync(self, line_num, text):
        if not self.available:
            print(f"[LCD Line {line_num}] {text}")
            return
        text = text.ljust(LCD_NUM_COLS)[:LCD_NUM_COLS]
        self._write_byte(LCD_LINE_ADDR[line_num], LCD_CMD)
        for char in text:
            self._write_byte(ord(char), LCD_CHR)

    def _show_recognition_sync(self, word, confidence):
        self._write_line_sync(0, "[ KSL Recognized ]")
        self._write_line_sync(1, f"> {word}")
        self._write_line_sync(2, f"  Conf: {confidence:.0%}")
        self._write_line_sync(3, "")

    def _show_buffer_sync(self, buffer_preview):
        self._write_line_sync(0, "[ Word Buffer    ]")
        self._write_line_sync(1, buffer_preview[:LCD_NUM_COLS])
        self._write_line_sync(2, "")
        self._write_line_sync(3, "  Hold for [완료]")

    def _show_sentence_sync(self, sentence):
        self._write_line_sync(0, "[ Generated      ]")
        words = sentence[:LCD_NUM_COLS * 3]
        for i in range(3):
            chunk = words[i * LCD_NUM_COLS:(i + 1) * LCD_NUM_COLS]
            self._write_line_sync(i + 1, chunk)

    # ── 하드웨어 직접 I/O (워커 스레드에서만 호출됨) ───────────────
    def _init_lcd(self):
        self._write_byte(0x33, LCD_CMD)
        self._write_byte(0x32, LCD_CMD)
        self._write_byte(0x06, LCD_CMD)
        self._write_byte(0x0C, LCD_CMD)
        self._write_byte(0x28, LCD_CMD)
        self._write_byte(0x01, LCD_CMD)
        time.sleep(0.05)

    def _write_byte(self, data, mode):
        high = mode | (data & 0xF0) | LCD_BACKLIGHT
        low = mode | ((data << 4) & 0xF0) | LCD_BACKLIGHT
        self.bus.write_byte(LCD_I2C_ADDRESS, high)
        self._toggle_enable(high)
        self.bus.write_byte(LCD_I2C_ADDRESS, low)
        self._toggle_enable(low)

    def _toggle_enable(self, data):
        time.sleep(0.0005)
        self.bus.write_byte(LCD_I2C_ADDRESS, data | ENABLE)
        time.sleep(0.0005)
        self.bus.write_byte(LCD_I2C_ADDRESS, data & ~ENABLE)
        time.sleep(0.0005)
