"""
lcd_display.py
I2C LCD 20x4 디스플레이 제어 모듈.
인식된 단어 버퍼와 생성된 문장을 LCD에 출력합니다.
"""

import smbus2
import time
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

    def _init_lcd(self):
        self._write_byte(0x33, LCD_CMD)
        self._write_byte(0x32, LCD_CMD)
        self._write_byte(0x06, LCD_CMD)
        self._write_byte(0x0C, LCD_CMD)
        self._write_byte(0x28, LCD_CMD)
        self._write_byte(0x01, LCD_CMD)
        time.sleep(0.05)

    def clear(self):
        if not self.available:
            return
        self._write_byte(0x01, LCD_CMD)
        time.sleep(0.05)

    def write_line(self, line_num, text):
        """
        특정 행(0~3)에 텍스트를 출력합니다.
        텍스트가 LCD_NUM_COLS를 초과하면 잘립니다.
        """
        if not self.available:
            print(f"[LCD Line {line_num}] {text}")
            return

        text = text.ljust(LCD_NUM_COLS)[:LCD_NUM_COLS]
        self._write_byte(LCD_LINE_ADDR[line_num], LCD_CMD)
        for char in text:
            self._write_byte(ord(char), LCD_CHR)

    def show_recognition(self, word, confidence):
        """수화 인식 결과 표시."""
        self.write_line(0, "[ KSL Recognized ]")
        self.write_line(1, f"> {word}")
        self.write_line(2, f"  Conf: {confidence:.0%}")
        self.write_line(3, "")

    def show_buffer(self, buffer_preview):
        """현재 단어 버퍼 미리보기 표시."""
        self.write_line(0, "[ Word Buffer    ]")
        self.write_line(1, buffer_preview[:LCD_NUM_COLS])
        self.write_line(2, "")
        self.write_line(3, "  Hold for [완료]")

    def show_sentence(self, sentence):
        """생성된 문장 표시 (최대 3행)."""
        self.write_line(0, "[ Generated      ]")
        words = sentence[:LCD_NUM_COLS * 3]
        for i in range(3):
            chunk = words[i * LCD_NUM_COLS:(i + 1) * LCD_NUM_COLS]
            self.write_line(i + 1, chunk)

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
