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
from config.settings import LCD_I2C_ADDRESS, LCD_NUM_COLS

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

        # 상단 줄(line 0) 상태 — 좌측 상태 태그 + 우측 현재 페르소나 표시용
        self._persona_label = ""
        self._last_state = "KSL"

        # 전광판(marquee) 스크롤 — 긴 문장을 한 줄에서 좌로 흘려 전체를 읽게 한다.
        # {text, pos, line} 또는 None. 워커 스레드에서만 접근.
        self._marquee = None
        self._tick = 0.4  # 스크롤 간격(초) — 큐가 비었을 때 한 칸 이동
        # 직전 버퍼 텍스트 — 동일 내용 재요청을 무시해(큐 비움) 전광판 스크롤을 허용한다.
        self._last_buffer = None

        # 비동기 I2C 쓰기 큐
        self._queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── 공개 API: 모두 큐에 작업 넣고 즉시 반환 ────────────────────
    def clear(self):
        self._last_buffer = None
        self._enqueue(self._clear_sync)

    def write_line(self, line_num, text):
        self._enqueue(lambda: self._write_line_sync(line_num, text))

    def show_recognition(self, word, confidence):
        self._last_buffer = None  # 인식 표시가 화면을 덮으므로 다음 버퍼는 다시 그린다
        self._enqueue(lambda: self._show_recognition_sync(word, confidence))

    def show_buffer(self, buffer_preview):
        # 동일 버퍼의 매-프레임 재요청은 큐를 채워 전광판 스크롤을 막는다(_scroll_tick은
        # 큐가 빌 때만 동작). 내용이 바뀔 때만 큐에 넣어 그 사이 워커가 스크롤하게 한다.
        if buffer_preview == self._last_buffer:
            return
        self._last_buffer = buffer_preview
        self._enqueue(lambda: self._show_buffer_sync(buffer_preview))

    def show_sentence(self, sentence):
        self._last_buffer = None
        self._enqueue(lambda: self._show_sentence_sync(sentence))

    def show_persona(self, label):
        """현재 출력 페르소나(예: 'Polite'/'Friendly')를 상단 줄 우측에 표시한다."""
        self._persona_label = label
        self._enqueue(lambda: self._write_line_sync(0, self._line0(self._last_state)))

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
                fn = self._queue.get(timeout=self._tick)
            except queue.Empty:
                self._scroll_tick()  # 큐가 비면 전광판 한 칸 이동
                continue
            try:
                fn()
            except Exception as e:
                print(f"[LCD] worker error: {e}")

    def _scroll_tick(self):
        """활성 marquee가 있으면 한 칸 좌로 이동해 다시 그린다."""
        m = self._marquee
        if not m:
            return
        full = m["text"]
        pos = m["pos"]
        window = (full + full)[pos:pos + LCD_NUM_COLS]
        self._write_line_sync(m["line"], window)
        m["pos"] = (pos + 1) % len(full)

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
        # HD44780은 ASCII만 렌더링 — 비ASCII(한글 등)는 '?'로 치환 (안전망).
        # 정상 경로에서는 호출자가 이미 영어 텍스트를 전달한다.
        text = "".join(c if ord(c) < 128 else "?" for c in text)
        text = text.ljust(LCD_NUM_COLS)[:LCD_NUM_COLS]
        self._write_byte(LCD_LINE_ADDR[line_num], LCD_CMD)
        for char in text:
            self._write_byte(ord(char), LCD_CHR)

    def _line0(self, state=None):
        """상단 줄 = 현재 페르소나만 표시(상태 문구 생략 — 글자 잘림 방지)."""
        if self._persona_label:
            return f"Style: {self._persona_label}"
        return "KSL-LLM-IoT"

    def _show_recognition_sync(self, word, confidence):
        self._marquee = None  # 인식 표시 중에는 스크롤 중지
        self._write_line_sync(0, self._line0("Recognized"))
        self._write_line_sync(1, f"> {word}")
        self._write_line_sync(2, f"  Conf: {confidence:.0%}")
        self._write_line_sync(3, "")

    def _show_buffer_sync(self, buffer_preview):
        self._write_line_sync(0, self._line0("Word Buffer"))
        self._write_line_sync(2, "")
        self._write_line_sync(3, " Press DONE button")
        if len(buffer_preview) <= LCD_NUM_COLS:
            # 한 줄에 들어가면 정적 표시
            self._marquee = None
            self._write_line_sync(1, buffer_preview)
        else:
            # 길면 전광판 스크롤 — 동일 버퍼 재요청을 무시(show_buffer)하므로 큐가 비고,
            # 워커가 _scroll_tick으로 좌로 흘려 누적 단어 전체를 보이게 한다.
            self._marquee = {"text": buffer_preview + "    ", "pos": 0, "line": 1}
            self._scroll_tick()  # 첫 프레임 즉시 표시

    def _show_sentence_sync(self, sentence):
        self._write_line_sync(0, self._line0("Generated"))
        self._write_line_sync(3, "")
        if len(sentence) <= LCD_NUM_COLS:
            # 한 줄에 들어가면 정적 표시(스크롤 불필요)
            self._marquee = None
            self._write_line_sync(1, sentence)
            self._write_line_sync(2, "")
        else:
            # 길면 전광판 스크롤 — 큐가 비는 동안 좌로 흐른다(완료 후 버퍼 비면 동작)
            self._marquee = {"text": sentence + "    ", "pos": 0, "line": 1}
            self._write_line_sync(2, "")
            self._scroll_tick()  # 첫 프레임 즉시 표시

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
