"""
button_input.py
물리 버튼 4개(완료 1 + 초기화 1 + 페르소나 2) 입력 처리.

배선: 각 버튼은 해당 GPIO 핀과 GND 사이에 연결한다 (내부 풀업 사용,
외부 저항 불필요, 눌림 = LOW). 핀 배치는 config/settings.py 참조.

설계 — 폴링 방식:
- RPi.GPIO의 add_event_detect는 최신 커널(Raspberry Pi OS Trixie)에서
  "Failed to add edge detection"으로 실패한다 (레거시 sysfs 이벤트와
  libgpiod 커널의 충돌 — RPi 실기 확인).
- 메인 루프(~30fps)가 매 프레임 poll()을 호출해 GPIO.input()으로
  하강 에지(HIGH→LOW)를 직접 검출한다. 루프 주기(~33ms)가 사람의
  버튼 누름 길이(>100ms)보다 충분히 짧아 놓침이 없다.
- 디바운스: 핀별 마지막 발화 시각 기록, BUTTON_DEBOUNCE_MS 이내 재발화 무시.
- gpio_module·time_fn 주입으로 PC 테스트에서 fake를 사용할 수 있다
  (CLAUDE.md §2.4 — RPi 전용 코드는 mock 처리).
"""

import time

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    BUTTON_COMPLETE_PIN, BUTTON_UNDO_PIN, BUTTON_PERSONA_PINS, BUTTON_DEBOUNCE_MS,
)

# poll()이 반환하는 이벤트 이름
EVENT_COMPLETE = "complete"  # 문장 완료 버튼
EVENT_UNDO = "undo"          # 초기화 버튼 — 버퍼 마지막 단어 제거


class ButtonInput:
    def __init__(self, gpio_module, time_fn=time.monotonic):
        """
        Args:
            gpio_module: RPi.GPIO 모듈 (또는 테스트용 fake).
                         호출 전에 setmode(BCM)가 완료되어 있어야 한다.
            time_fn: 현재 시각(초) 함수 — 테스트에서 가상 시계 주입용.
        """
        self._gpio = gpio_module
        self._time = time_fn
        self._pin_to_event = {
            BUTTON_COMPLETE_PIN: EVENT_COMPLETE,
            BUTTON_UNDO_PIN: EVENT_UNDO,
        }
        for persona, pin in BUTTON_PERSONA_PINS.items():
            self._pin_to_event[pin] = persona

        self._last_state = {}
        self._last_fire = {}
        for pin in self._pin_to_event:
            self._gpio.setup(pin, self._gpio.IN, pull_up_down=self._gpio.PUD_UP)
            self._last_state[pin] = 1  # 풀업 기본 = HIGH (안 눌림)
            self._last_fire[pin] = float("-inf")

    def poll(self):
        """버튼 하강 에지를 검출해 이벤트 1건을 반환, 없으면 None.

        한 번에 1건만 반환한다 — 같은 호출에서 처리하지 않은 핀의 상태는
        갱신하지 않으므로 다음 poll()에서 누락 없이 검출된다.

        Returns:
            EVENT_COMPLETE | EVENT_UNDO | 페르소나 이름("정중"/"친근") | None
        """
        now = self._time()
        for pin, event in self._pin_to_event.items():
            state = self._gpio.input(pin)
            prev = self._last_state[pin]
            self._last_state[pin] = state
            if prev and not state:  # HIGH → LOW = 눌림
                if (now - self._last_fire[pin]) * 1000.0 >= BUTTON_DEBOUNCE_MS:
                    self._last_fire[pin] = now
                    return event
        return None
