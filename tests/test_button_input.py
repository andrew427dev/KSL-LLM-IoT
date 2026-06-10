"""
test_button_input.py
물리 버튼 입력(src/button_input.py) 검증 — RPi.GPIO 없이 fake GPIO 주입.

로컬 실행:
    python tests/test_button_input.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    BUTTON_COMPLETE_PIN, BUTTON_PERSONA_PINS, BUTTON_DEBOUNCE_MS, BUZZER_PIN,
)
from src.button_input import ButtonInput, EVENT_COMPLETE


class FakeGPIO:
    """RPi.GPIO 흉내 — setup/add_event_detect 기록 + 콜백 수동 발화."""
    IN = "IN"
    PUD_UP = "PUD_UP"
    FALLING = "FALLING"

    def __init__(self):
        self.setups = {}
        self.callbacks = {}
        self.bouncetimes = {}

    def setup(self, pin, direction, pull_up_down=None):
        self.setups[pin] = (direction, pull_up_down)

    def add_event_detect(self, pin, edge, callback, bouncetime):
        assert edge == self.FALLING
        self.callbacks[pin] = callback
        self.bouncetimes[pin] = bouncetime

    def press(self, pin):
        """버튼 눌림 시뮬레이션 — RPi.GPIO처럼 핀 번호를 콜백에 전달."""
        self.callbacks[pin](pin)


def test_pins_configured_with_pullup():
    """버튼 4핀이 모두 입력+풀업으로 설정되고 디바운스가 적용돼야 한다."""
    gpio = FakeGPIO()
    ButtonInput(gpio)
    expected_pins = {BUTTON_COMPLETE_PIN, *BUTTON_PERSONA_PINS.values()}
    assert set(gpio.setups) == expected_pins
    for pin in expected_pins:
        assert gpio.setups[pin] == (FakeGPIO.IN, FakeGPIO.PUD_UP)
        assert gpio.bouncetimes[pin] == BUTTON_DEBOUNCE_MS
    print("[PASS] test_pins_configured_with_pullup")


def test_press_events_routed():
    """완료 버튼 → EVENT_COMPLETE, 페르소나 버튼 → 페르소나 이름."""
    gpio = FakeGPIO()
    buttons = ButtonInput(gpio)

    gpio.press(BUTTON_COMPLETE_PIN)
    assert buttons.poll() == EVENT_COMPLETE

    for persona, pin in BUTTON_PERSONA_PINS.items():
        gpio.press(pin)
        assert buttons.poll() == persona

    assert buttons.poll() is None  # 큐 소진
    print("[PASS] test_press_events_routed")


def test_event_order_preserved():
    """연속 입력은 누른 순서대로 회수돼야 한다 (FIFO)."""
    gpio = FakeGPIO()
    buttons = ButtonInput(gpio)
    order = [BUTTON_PERSONA_PINS["간단"], BUTTON_COMPLETE_PIN,
             BUTTON_PERSONA_PINS["정중"]]
    for pin in order:
        gpio.press(pin)
    assert buttons.poll() == "간단"
    assert buttons.poll() == EVENT_COMPLETE
    assert buttons.poll() == "정중"
    print("[PASS] test_event_order_preserved")


def test_no_pin_conflicts():
    """버튼 핀들이 서로·부저·I2C(2,3)와 겹치지 않아야 한다."""
    pins = [BUTTON_COMPLETE_PIN, *BUTTON_PERSONA_PINS.values()]
    assert len(pins) == len(set(pins)), "버튼 핀 중복"
    reserved = {BUZZER_PIN, 2, 3}  # 부저, I2C SDA/SCL
    assert not (set(pins) & reserved), f"예약 핀과 충돌: {set(pins) & reserved}"
    print("[PASS] test_no_pin_conflicts")


if __name__ == "__main__":
    test_pins_configured_with_pullup()
    test_press_events_routed()
    test_event_order_preserved()
    test_no_pin_conflicts()
    print("\nAll tests done.")
