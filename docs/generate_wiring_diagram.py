"""
generate_wiring_diagram.py
KSL-LLM-IoT 하드웨어 배선도를 생성합니다.
출력: docs/wiring_diagram.png

Usage:
    python docs/generate_wiring_diagram.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "wiring_diagram.png")

# ── 색상 팔레트 ─────────────────────────────────────────
C_RPI    = "#c0392b"   # Raspberry Pi — 빨강
C_LCD    = "#2980b9"   # LCD — 파랑
C_BUZZ   = "#27ae60"   # 부저 — 초록
C_CAM    = "#8e44ad"   # 카메라 — 보라
C_SPK    = "#e67e22"   # 스피커 — 주황
C_BTN    = "#16a085"   # 푸시 버튼 — 청록
C_LED    = "#f1c40f"   # 상태 LED — 노랑
C_WIRE_5V  = "#e74c3c" # 5V 선
C_WIRE_GND = "#2c3e50" # GND 선
C_WIRE_SIG = "#f39c12" # 신호 선
C_WIRE_I2C = "#3498db" # I2C 선
C_BG     = "#fafafa"


def draw_box(ax, x, y, w, h, label, color, fontsize=10, text_color="white"):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="white",
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=4)


def draw_wire(ax, x1, y1, x2, y2, color, label="", lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=7, color=color,
                ha='center', va='bottom', zorder=5,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.8))


def main():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title("KSL-LLM-IoT  Hardware Wiring Diagram\n"
                 "Raspberry Pi 4B — I2C LCD 20×4 — USB Webcam — Buzzer — "
                 "Push Buttons ×4 — Status LED — Speaker (USB power + 3.5mm audio)",
                 fontsize=13, fontweight='bold', pad=14, color='#2c3e50')

    # ── Raspberry Pi 4B (중앙) ───────────────────────────
    rpi_x, rpi_y, rpi_w, rpi_h = 4.8, 3.2, 4.4, 3.0
    draw_box(ax, rpi_x, rpi_y, rpi_w, rpi_h, "Raspberry Pi 4B", C_RPI, fontsize=12)

    # 핀 레이블 (RPi 내부)
    pins = [
        ("GPIO 2 (SDA)", rpi_x + 0.15, rpi_y + 2.3),
        ("GPIO 3 (SCL)", rpi_x + 0.15, rpi_y + 1.9),
        ("GPIO 17",      rpi_x + 0.15, rpi_y + 1.5),
        ("GPIO 22",      rpi_x + 0.15, rpi_y + 0.7),
        ("GND (Pin 25)", rpi_x + 0.15, rpi_y + 0.35),
        ("5V  (Pin 2)",  rpi_x + 0.15, rpi_y + 2.7),
        ("GND (Pin 6)",  rpi_x + 0.15, rpi_y + 1.1),
        ("USB Port",     rpi_x + rpi_w - 1.2, rpi_y + 2.5),
        ("3.5mm Jack",   rpi_x + rpi_w - 1.2, rpi_y + 0.35),
    ]
    for text, px, py in pins:
        ax.text(px, py, f"• {text}", fontsize=7.5, color='white', zorder=5)

    # ── I2C LCD 20x4 (왼쪽) ─────────────────────────────
    lcd_x, lcd_y = 0.5, 4.5
    draw_box(ax, lcd_x, lcd_y, 3.0, 1.8, "I2C LCD 20×4\n(0x27)", C_LCD, fontsize=10)

    # LCD 배선
    draw_wire(ax, lcd_x + 3.0, lcd_y + 1.5, rpi_x, rpi_y + 2.7,
              C_WIRE_5V, "5V")
    draw_wire(ax, lcd_x + 3.0, lcd_y + 1.2, rpi_x, rpi_y + 2.3,
              C_WIRE_I2C, "SDA")
    draw_wire(ax, lcd_x + 3.0, lcd_y + 0.9, rpi_x, rpi_y + 1.9,
              C_WIRE_I2C, "SCL")
    draw_wire(ax, lcd_x + 3.0, lcd_y + 0.5, rpi_x, rpi_y + 1.1,
              C_WIRE_GND, "GND")

    # ── 능동 부저 (왼쪽 하단) ───────────────────────────
    bz_x, bz_y = 0.5, 1.5
    bz_w, bz_h = 3.0, 1.6
    draw_box(ax, bz_x, bz_y, bz_w, bz_h, "Active Buzzer", C_BUZZ, fontsize=10)
    ax.text(bz_x + bz_w / 2, bz_y + 0.28, "+ = Signal     – = GND",
            fontsize=7, color='white', ha='center', va='center', zorder=5)

    draw_wire(ax, bz_x + bz_w, bz_y + 1.1, rpi_x, rpi_y + 1.5,
              C_WIRE_SIG, "GPIO 17")
    draw_wire(ax, bz_x + bz_w, bz_y + 0.4, rpi_x, rpi_y + 1.1,
              C_WIRE_GND, "GND")

    # ── 상태 LED (버저 동기, 시각 피드백 — 농인 접근성) ─────
    led_x, led_y = 0.5, 3.35
    led_w, led_h = 3.0, 0.85
    draw_box(ax, led_x, led_y, led_w, led_h, "Status LED",
             C_LED, fontsize=8.5, text_color="#2c3e50")
    # 신호: LED anode(+) → GPIO22 (직렬 저항 220~330Ω 권장), cathode(−) → GND
    draw_wire(ax, led_x + led_w, led_y + 0.55, rpi_x, rpi_y + 0.7,
              C_WIRE_SIG, "GPIO 22")
    draw_wire(ax, led_x + led_w, led_y + 0.2, rpi_x, rpi_y + 0.35,
              C_WIRE_GND, "GND Pin 25")

    # ── 푸시 버튼 ×4 (하단 중앙) ─────────────────────────
    # 각 버튼: 한쪽 다리 → 해당 GPIO, 다른 다리 → GND(물리 6, 실측 2026-06-12).
    # 내부 풀업 사용 — 외부 저항 불필요, 눌림 = LOW.
    btn_x, btn_y = 4.9, 0.5
    draw_box(ax, btn_x, btn_y, 4.2, 1.3,
             "Push Buttons ×4  (other leg → GND Pin 6)\n"
             "Complete=GPIO5(29)  Undo=GPIO6(31)\n"
             "Polite=GPIO13(33)  Friendly=GPIO19(35)",
             C_BTN, fontsize=8)
    for i, gpio_label in enumerate(["G5", "G6", "G13", "G19"]):
        wx = btn_x + 1.1 + i * 0.9
        draw_wire(ax, wx, btn_y + 1.3, wx, rpi_y, C_WIRE_SIG, gpio_label)
    # 공통 GND — 4개 버튼의 반대쪽 다리가 모두 GND(Pin 6)로 연결되어야 동작
    draw_wire(ax, btn_x + 0.25, btn_y + 1.3, rpi_x + 0.25, rpi_y,
              C_WIRE_GND, "GND Pin 6")

    # ── USB 웹캠 (상단) ──────────────────────────────────
    cam_x, cam_y = 5.3, 7.2
    draw_box(ax, cam_x, cam_y, 3.4, 1.2, "USB Webcam", C_CAM, fontsize=10)
    draw_wire(ax, cam_x + 1.7, cam_y, rpi_x + rpi_w / 2, rpi_y + rpi_h,
              C_WIRE_SIG, "USB 3.0")

    # ── Speaker (오른쪽) — USB 5V 전원 + 3.5mm 오디오 (둘 다 필요) ──
    spk_x, spk_y = 10.2, 4.3
    spk_w, spk_h = 3.0, 1.6
    draw_box(ax, spk_x, spk_y, spk_w, spk_h,
             "Active Speaker\nUSB = power\n3.5mm = audio\n(both required)",
             C_SPK, fontsize=8.5)
    # USB 전원선 → RPi USB 포트 (USB 2.0)
    draw_wire(ax, spk_x, spk_y + 1.1, rpi_x + rpi_w, rpi_y + 2.5,
              C_WIRE_5V, "USB 2.0")
    # 3.5mm 오디오선 → RPi 3.5mm 잭
    draw_wire(ax, spk_x, spk_y + 0.4, rpi_x + rpi_w, rpi_y + 0.4,
              C_WIRE_SIG, "3.5mm audio")

    # ── 범례 ────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_WIRE_5V,  label="5V Power"),
        mpatches.Patch(color=C_WIRE_GND, label="GND"),
        mpatches.Patch(color=C_WIRE_I2C, label="I2C (SDA/SCL)"),
        mpatches.Patch(color=C_WIRE_SIG, label="Signal / Data"),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              fontsize=9, framealpha=0.9, title="Wire Colors", title_fontsize=9)

    # ── GPIO 요약 표 ─────────────────────────────────────
    table_x, table_y = 9.8, 1.5
    ax.text(table_x, table_y + 2.0, "GPIO Pin Summary",
            fontsize=9, fontweight='bold', color='#2c3e50')
    rows = [
        ("GPIO 2 (SDA)", "I2C LCD — Data"),
        ("GPIO 3 (SCL)", "I2C LCD — Clock"),
        ("GPIO 17",      "Buzzer (+)"),
        ("GPIO 22",      "Status LED (+, via resistor)"),
        ("GPIO 5/6/13/19", "Buttons: complete/undo/persona x2"),
        ("5V  Pin 2",    "LCD VCC"),
        ("GND Pin 6",    "LCD/Buzzer (–) / Button legs"),
        ("GND Pin 25",   "Status LED (–)"),
        ("USB Port",     "Webcam / Speaker 5V power"),
        ("3.5mm Jack",   "Speaker audio"),
    ]
    for i, (pin, desc) in enumerate(rows):
        y_pos = table_y + 1.7 - i * 0.2
        ax.text(table_x, y_pos, f"{pin:<16} {desc}", fontsize=7,
                color='#2c3e50', family='monospace')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print(f"[Wiring Diagram] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
