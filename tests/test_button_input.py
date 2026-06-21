"""
test_button_input.py
물리 버튼 입력(src/button_input.py, 폴링 방식) 검증 — RPi.GPIO 없이
fake GPIO·가상 시계 주입.

로컬 실행:
    python tests/test_button_input.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    BUTTON_COMPLETE_PIN, BUTTON_UNDO_PIN, BUTTON_PERSONA_PINS,
    BUTTON_DEBOUNCE_MS, BUZZER_PIN, LED_PIN,
)
from src.button_input import ButtonInput, EVENT_COMPLETE, EVENT_UNDO


class FakeGPIO:
    """RPi.GPIO 흉내 — setup 기록 + 핀 레벨 시뮬레이션 (풀업 기본 HIGH)."""
    IN = "IN"
    PUD_UP = "PUD_UP"

    def __init__(self):
        self.setups = {}
        self.levels = {}

    def setup(self, pin, direction, pull_up_down=None):
        self.setups[pin] = (direction, pull_up_down)
        self.levels[pin] = 1  # 풀업 — 안 눌림

    def input(self, pin):
        return self.levels[pin]

    def press(self, pin):
        self.levels[pin] = 0

    def release(self, pin):
        self.levels[pin] = 1


class FakeClock:
    """가상 시계 — 디바운스 검증용."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += sec


def make():
    gpio, clock = FakeGPIO(), FakeClock()
    return gpio, clock, ButtonInput(gpio, time_fn=clock)


def test_pins_configured_with_pullup():
    """버튼 4핀이 모두 입력+풀업으로 설정돼야 한다."""
    gpio, _clock, _buttons = make()
    expected_pins = {BUTTON_COMPLETE_PIN, BUTTON_UNDO_PIN, *BUTTON_PERSONA_PINS.values()}
    assert set(gpio.setups) == expected_pins
    for pin in expected_pins:
        assert gpio.setups[pin] == (FakeGPIO.IN, FakeGPIO.PUD_UP)
    print("[PASS] test_pins_configured_with_pullup")


def test_falling_edge_routed():
    """완료 버튼 → EVENT_COMPLETE, 페르소나 버튼 → 페르소나 이름.
    누른 채 유지하면 재발화하지 않고, 뗐다 다시 누르면 재검출."""
    gpio, clock, buttons = make()

    assert buttons.poll() is None  # 초기 상태 — 이벤트 없음

    gpio.press(BUTTON_COMPLETE_PIN)
    assert buttons.poll() == EVENT_COMPLETE
    assert buttons.poll() is None, "누른 채 유지 — 레벨이 아니라 에지 검출이어야 한다"

    gpio.release(BUTTON_COMPLETE_PIN)
    assert buttons.poll() is None

    for persona, pin in BUTTON_PERSONA_PINS.items():
        clock.advance(1.0)
        gpio.press(pin)
        assert buttons.poll() == persona
        gpio.release(pin)
        buttons.poll()
    print("[PASS] test_falling_edge_routed")


def test_debounce():
    """디바운스 시간 내의 재눌림(채터링)은 무시, 경과 후에는 재검출."""
    gpio, clock, buttons = make()
    pin = BUTTON_COMPLETE_PIN

    gpio.press(pin)
    assert buttons.poll() == EVENT_COMPLETE

    # 채터링: 디바운스 시간 내 뗐다 다시 눌림
    gpio.release(pin); buttons.poll()
    clock.advance(BUTTON_DEBOUNCE_MS / 1000.0 * 0.5)
    gpio.press(pin)
    assert buttons.poll() is None, "디바운스 내 재눌림은 무시돼야 한다"

    # 디바운스 경과 후 정상 재검출
    gpio.release(pin); buttons.poll()
    clock.advance(BUTTON_DEBOUNCE_MS / 1000.0 + 0.01)
    gpio.press(pin)
    assert buttons.poll() == EVENT_COMPLETE
    print("[PASS] test_debounce")


def test_simultaneous_press_no_loss():
    """두 버튼 동시 눌림 — 한 poll에 1건씩, 누락 없이 순차 반환."""
    gpio, clock, buttons = make()
    gpio.press(BUTTON_COMPLETE_PIN)
    gpio.press(BUTTON_PERSONA_PINS["친근"])
    first = buttons.poll()
    second = buttons.poll()
    assert {first, second} == {EVENT_COMPLETE, "친근"}
    assert buttons.poll() is None
    print("[PASS] test_simultaneous_press_no_loss")


def test_undo_button_routed():
    """초기화 버튼 → EVENT_UNDO."""
    gpio, clock, buttons = make()
    gpio.press(BUTTON_UNDO_PIN)
    assert buttons.poll() == EVENT_UNDO
    print("[PASS] test_undo_button_routed")


def test_no_pin_conflicts():
    """버튼 핀들이 서로·부저·LED·I2C(2,3)와 겹치지 않아야 한다."""
    pins = [BUTTON_COMPLETE_PIN, BUTTON_UNDO_PIN, *BUTTON_PERSONA_PINS.values()]
    assert len(pins) == len(set(pins)), "버튼 핀 중복"
    reserved = {BUZZER_PIN, LED_PIN, 2, 3}  # 부저, LED, I2C SDA/SCL
    assert not (set(pins) & reserved), f"예약 핀과 충돌: {set(pins) & reserved}"
    print("[PASS] test_no_pin_conflicts")


if __name__ == "__main__":
    test_pins_configured_with_pullup()
    test_falling_edge_routed()
    test_debounce()
    test_simultaneous_press_no_loss()
    test_undo_button_routed()
    test_no_pin_conflicts()
    print("\nAll tests done.")
