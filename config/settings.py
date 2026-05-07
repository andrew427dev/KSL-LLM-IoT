"""
settings.py
Global configuration for KSL-LLM-IoT system.
All sensitive values (API keys) must be set in .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Gemini API ──────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_TOKENS = 100

GEMINI_SYSTEM_PROMPT = """
너는 한국 수화(KSL) 번역 보조 AI야.
사용자가 수화로 표현한 단어 목록을 받으면,
자연스러운 현대 한국어 문장 1~2개로 변환해줘.
규칙:
- 존댓말 사용
- 문장은 간결하고 명확하게
- 단어 순서가 문법적으로 이상해도 의미를 파악해서 자연스럽게 변환
- 번역 결과만 출력, 설명 없이
"""

# ── Model ───────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/ksl_model.tflite")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 30))  # frames
NUM_LANDMARKS = 21
NUM_AXES = 3  # x, y, z
INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_LANDMARKS * NUM_AXES)  # (30, 63)

# ── KSL Word Labels ─────────────────────────────────────
KSL_LABELS = [
    "안녕", "감사합니다", "미안합니다", "반갑습니다",
    "나", "당신", "우리",
    "네", "아니오", "좋다", "싫다", "맞다",
    "먹다", "마시다", "가다", "오다", "앉다", "서다", "자다",
    "배고프다", "목마르다", "아프다", "피곤하다", "행복하다",
    "도와주세요", "주세요", "기다리세요",
    "화장실", "얼마예요",
    "완료"  # trigger word to send to Gemini
]

TRIGGER_WORD = "완료"
SILENCE_TRIGGER_SEC = float(os.getenv("SILENCE_TRIGGER_SEC", 3.0))
DUPLICATE_FILTER_SEC = 1.5  # prevent same word repeated within N seconds

# ── Hardware: LCD ───────────────────────────────────────
LCD_I2C_ADDRESS = int(os.getenv("LCD_I2C_ADDRESS", "0x27"), 16)
LCD_NUM_COLS = int(os.getenv("LCD_NUM_COLS", 20))
LCD_NUM_ROWS = int(os.getenv("LCD_NUM_ROWS", 4))

# ── Hardware: GPIO ──────────────────────────────────────
BUZZER_PIN = int(os.getenv("BUZZER_PIN", 17))

# ── TTS ─────────────────────────────────────────────────
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "ko")
TTS_OFFLINE_MODE = os.getenv("TTS_OFFLINE_MODE", "false").lower() == "true"

# ── Camera ──────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", 640))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", 480))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", 30))
