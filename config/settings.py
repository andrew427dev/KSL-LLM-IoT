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
- 문장은 간결하고 명확하게
- 단어 순서가 문법적으로 이상해도 의미를 파악해서 자연스럽게 변환
- 번역 결과만 출력, 설명 없이
"""

# ── 문장 생성 페르소나 ──────────────────────────────────
# 사용자가 기분·상황에 따라 출력 문체를 선택한다. 위 기본 프롬프트에
# 아래 문체 지시문이 덧붙는다. 설정: .env SENTENCE_PERSONA,
# GUI 모드에서는 'p' 키로 실행 중 순환 변경.
SENTENCE_PERSONAS = {
    "정중": "- 문체: 격식 있는 존댓말(-습니다체)로 정중하게 표현한다.",
    "친근": "- 문체: 부드러운 해요체로 친근하고 따뜻하게 표현한다.",
    "간단": "- 문체: 핵심만 남겨 짧고 간결하게 표현한다. 한 문장으로.",
}
SENTENCE_PERSONA = os.getenv("SENTENCE_PERSONA", "정중")
if SENTENCE_PERSONA not in SENTENCE_PERSONAS:
    SENTENCE_PERSONA = "정중"

# ── Model ───────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/ksl_model.tflite")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 30))  # frames
NUM_HANDS = 2          # KSL는 양손 언어 — LEFT/RIGHT 두 손 입력 고정
NUM_LANDMARKS = 21     # per hand
NUM_AXES = 3           # x, y, z
WRIST_REL_VEC = 3      # 우손목 − 좌손목 (intra-hand scale 평균으로 정규화된 3D)
PRESENCE_FLAG = 2      # [left_present, right_present]

# 한 프레임 = [LEFT_63 | RIGHT_63 | wrist_vec_3 | presence_2] = 131.
# 각 손은 손목 상대좌표를 intra-hand scale(‖landmark9−landmark0‖)로 나눠 정규화.
# 미감지 손은 zero + presence=0. 양손 모두 미감지면 프레임 자체를 시퀀스에서 제외.
LEFT_SLOT_START = 0
RIGHT_SLOT_START = NUM_LANDMARKS * NUM_AXES                  # 63
WRIST_VEC_START = NUM_HANDS * NUM_LANDMARKS * NUM_AXES       # 126
PRESENCE_FLAG_START = WRIST_VEC_START + WRIST_REL_VEC        # 129
FEATURE_DIM = PRESENCE_FLAG_START + PRESENCE_FLAG            # 131
INPUT_SHAPE = (SEQUENCE_LENGTH, FEATURE_DIM)                 # (30, 131)

# handedness 신뢰도 임계값 — 미달 시 라벨 대신 x좌표 fallback으로 슬롯 배정.
# PR #7 실기 검증에서 0.7로 확정. 수집 데이터 분포에 따라 0.6~0.8 재조정 가능.
HANDEDNESS_SCORE_THRESHOLD = float(os.getenv("HANDEDNESS_SCORE_THRESHOLD", 0.7))

# 연속 N프레임 양손 미검출 시 분류기 시퀀스 버퍼를 비운다 (stale 추론 방지).
NO_HAND_RESET_FRAMES = int(os.getenv("NO_HAND_RESET_FRAMES", 10))

# ── KSL Word Labels ─────────────────────────────────────
# AI Hub 수어 데이터셋(3,000단어)의 사전 표제어와 정확히 일치하는 30단어
# (2026-06-10 확정). 어근 매칭의 의미 오염을 피하기 위해 convert_aihub.py는
# --exact 모드로 변환한다. 라벨 변경 = 데이터 재변환 + 모델 재학습 필요.
KSL_LABELS = [
    # 인칭·응답
    "나", "당신", "좋다", "싫다", "맞다",
    # 동사
    "가다", "오다", "서다", "자다", "주다",
    # 상태
    "배고프다", "목마르다", "아프다", "피곤하다",
    "춥다", "덥다", "슬프다", "화나다",
    # 감정·의사소통
    "행복", "감사", "부탁", "돕다",
    # 생활 명사
    "밥", "병원", "의사", "엄마", "가족", "친구", "얼마",
    # 시스템 트리거 — 항상 마지막 원소 유지 (CLAUDE.md §2.3)
    "완료"
]

# 학습 실험 전용 오버라이드 — 쉼표 구분 단어 목록으로 KSL_LABELS를 대체한다.
# 임의 어휘로 변환·학습 파이프라인을 검증할 때 사용 (예: 서버 종단 검증).
# 운용 배포에서는 사용하지 않는다 — TRIGGER_WORD가 목록에 없으면 문장 트리거는
# 무음 타이머(SILENCE_TRIGGER_SEC)로만 동작한다.
_labels_override = os.getenv("KSL_LABELS_OVERRIDE", "").strip()
if _labels_override:
    KSL_LABELS = [w.strip() for w in _labels_override.split(",") if w.strip()]

TRIGGER_WORD = "완료"
# 무동작 자동 문장화(초). 0 이하 = 비활성. 기본 비활성 — 인식율 의존
# 레이턴시 오차가 커서 물리 버튼(BUTTON_COMPLETE_PIN)으로 대체 (2026-06-10).
# 복원하려면 .env에 SILENCE_TRIGGER_SEC=3.0
SILENCE_TRIGGER_SEC = float(os.getenv("SILENCE_TRIGGER_SEC", 0.0))
DUPLICATE_FILTER_SEC = 1.5  # prevent same word repeated within N seconds

# ── Hardware: LCD ───────────────────────────────────────
LCD_I2C_ADDRESS = int(os.getenv("LCD_I2C_ADDRESS", "0x27"), 16)
LCD_NUM_COLS = int(os.getenv("LCD_NUM_COLS", 20))
LCD_NUM_ROWS = int(os.getenv("LCD_NUM_ROWS", 4))

# ── Hardware: GPIO ──────────────────────────────────────
BUZZER_PIN = int(os.getenv("BUZZER_PIN", 17))

# 물리 버튼 (BCM). 내부 풀업 — 버튼은 해당 핀과 GND 사이에 연결, 눌림=LOW.
# 물리핀: GPIO5=29, GPIO6=31, GPIO13=33, GPIO19=35 (GND: 30, 34 인접).
# 문장 완료 버튼: 버퍼 단어를 즉시 Gemini 문장으로 변환.
# 페르소나 버튼 3개: 문체 즉시 전환 (비프 1/2/3회 피드백).
BUTTON_COMPLETE_PIN = int(os.getenv("BUTTON_COMPLETE_PIN", 5))
BUTTON_PERSONA_PINS = {
    "정중": int(os.getenv("BUTTON_PERSONA_POLITE_PIN", 6)),
    "친근": int(os.getenv("BUTTON_PERSONA_FRIENDLY_PIN", 13)),
    "간단": int(os.getenv("BUTTON_PERSONA_BRIEF_PIN", 19)),
}
BUTTON_DEBOUNCE_MS = int(os.getenv("BUTTON_DEBOUNCE_MS", 300))

# ── TTS ─────────────────────────────────────────────────
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "ko")
TTS_OFFLINE_MODE = os.getenv("TTS_OFFLINE_MODE", "false").lower() == "true"

# ── Camera ──────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", 640))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", 480))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", 30))
