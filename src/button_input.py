"""
button_input.py
물리 버튼 4개(문장 완료 1 + 페르소나 3) 입력 처리.

배선: 각 버튼은 해당 GPIO 핀과 GND 사이에 연결한다 (내부 풀업 사용,
외부 저항 불필요, 눌림 = LOW). 핀 배치는 config/settings.py 참조.

설계:
- RPi.GPIO의 add_event_detect 콜백은 별도 스레드에서 호출되므로,
  콜백에서는 thread-safe 큐에 이벤트 이름만 넣고 실제 처리(LCD·
  SentenceBuilder 조작)는 메인 루프가 poll()로 회수해 수행한다.
- 디바운스는 RPi.GPIO bouncetime(BUTTON_DEBOUNCE_MS)에 위임한다.
- gpio_module 주입으로 PC 테스트에서 fake GPIO를 사용할 수 있다
  (CLAUDE.md §2.4 — RPi 전용 코드는 mock 처리).
"""

import queue

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    BUTTON_COMPLETE_PIN, BUTTON_PERSONA_PINS, BUTTON_DEBOUNCE_MS,
)

# poll()이 반환하는 이벤트 이름
EVENT_COMPLETE = "complete"  # 문장 완료 버튼


class ButtonInput:
    def __init__(self, gpio_module):
        """
        Args:
            gpio_module: RPi.GPIO 모듈 (또는 테스트용 fake).
                         호출 전에 setmode(BCM)가 완료되어 있어야 한다.
        """
        self._gpio = gpio_module
        self._events = queue.Queue()
        self._pin_to_event = {BUTTON_COMPLETE_PIN: EVENT_COMPLETE}
        for persona, pin in BUTTON_PERSONA_PINS.items():
            self._pin_to_event[pin] = persona

        for pin in self._pin_to_event:
            self._gpio.setup(pin, self._gpio.IN, pull_up_down=self._gpio.PUD_UP)
            self._gpio.add_event_detect(
                pin, self._gpio.FALLING,
                callback=self._on_press,
                bouncetime=BUTTON_DEBOUNCE_MS,
            )

    def _on_press(self, pin):
        """GPIO 콜백 스레드 — 큐 적재만 수행 (본 처리는 메인 루프)."""
        event = self._pin_to_event.get(pin)
        if event is not None:
            self._events.put(event)

    def poll(self):
        """대기 중인 버튼 이벤트 1건을 반환, 없으면 None.

        Returns:
            EVENT_COMPLETE | 페르소나 이름("정중"/"친근"/"간단") | None
        """
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None
