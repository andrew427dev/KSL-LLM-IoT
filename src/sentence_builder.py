"""
sentence_builder.py
인식된 수화 단어를 버퍼에 축적하고, 트리거 조건 충족 시 Gemini API를
비동기로 호출해 자연어 문장을 생성한다.

Gemini API 호출(1~3초)이 메인 인식 루프를 블록하지 않도록 워커 스레드에서
처리하며, 완료 결과는 큐로 전달한다. 메인 루프는 매 프레임 poll_sentence()로
완료 여부를 폴링한다.
"""

import time
import queue
import threading

from google import genai
from google.genai import types

from config.settings import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS,
    GEMINI_SYSTEM_PROMPT, TRIGGER_WORD, SILENCE_TRIGGER_SEC,
    SENTENCE_PERSONAS, SENTENCE_PERSONA,
)


class SentenceBuilder:
    def __init__(self):
        # GEMINI_API_KEY가 비어있으면 오프라인 모드로 동작. 단어들을
        # 공백으로 이어 붙여 fallback 문장을 만든다. API 키가 있으면
        # 정상 LLM 경로.
        self.persona = SENTENCE_PERSONA
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._offline = False
            self._rebuild_config()
        else:
            print("[SentenceBuilder] No GEMINI_API_KEY in .env. "
                  "OFFLINE MODE: sentences = space-joined word list.")
            self._client = None
            self._gen_config = None
            self._offline = True

        self.word_buffer = []
        self._last_word_time = time.time()
        self._completed = queue.Queue()
        self._inflight = False
        self._inflight_lock = threading.Lock()

    def set_persona(self, persona):
        """문장 문체 페르소나를 변경한다 (정중/친근/간단).

        등록되지 않은 이름이면 변경하지 않고 False를 반환한다.
        오프라인 모드에서는 이름만 보관한다 (출력은 단어 나열이므로 무효과).
        """
        if persona not in SENTENCE_PERSONAS:
            return False
        self.persona = persona
        if not self._offline:
            self._rebuild_config()
        return True

    def _rebuild_config(self):
        """현재 페르소나 지시문을 덧붙인 생성 config를 재구성한다.

        워커 스레드는 self._gen_config 참조를 읽기만 하므로
        참조 교체(원자적)로 동시성 문제가 없다.
        """
        self._gen_config = types.GenerateContentConfig(
            system_instruction=(
                GEMINI_SYSTEM_PROMPT + "\n" + SENTENCE_PERSONAS[self.persona]
            ),
            max_output_tokens=GEMINI_MAX_TOKENS,
            temperature=0.3,
        )

    def add_word(self, word):
        """
        인식된 단어를 버퍼에 추가한다.
        TRIGGER_WORD("완료")는 즉시 비동기 문장 생성을 시작한다.
        반환값 없음 — 완료된 문장은 poll_sentence()로 회수한다.
        """
        if word == TRIGGER_WORD:
            self._trigger_async()
            return

        self.word_buffer.append(word)
        self._last_word_time = time.time()

    def trigger_sentence(self):
        """버퍼 단어로 즉시 비동기 문장 생성을 시작한다 (물리 버튼·키 입력용).

        버퍼가 비어 있거나 이미 생성 중이면 시작하지 않고 False를 반환한다.
        """
        with self._inflight_lock:
            can_start = bool(self.word_buffer) and not self._inflight
        if can_start:
            self._trigger_async()
        return can_start

    def check_silence_trigger(self):
        """
        마지막 단어 인식 후 SILENCE_TRIGGER_SEC 이상 경과하면
        비동기 문장 생성을 시작한다. SILENCE_TRIGGER_SEC가 0 이하면
        비활성 — 물리 완료 버튼(BUTTON_COMPLETE_PIN)이 트리거를 담당한다.
        """
        if SILENCE_TRIGGER_SEC <= 0:
            return
        if not self.word_buffer:
            return
        if time.time() - self._last_word_time >= SILENCE_TRIGGER_SEC:
            self._trigger_async()

    def poll_sentence(self):
        """
        완료된 문장이 있으면 반환, 없으면 None.
        메인 루프가 매 프레임 호출한다.
        """
        try:
            return self._completed.get_nowait()
        except queue.Empty:
            return None

    def get_buffer_preview(self):
        """LCD 실시간 표시용 현재 버퍼 내용."""
        return " | ".join(self.word_buffer) if self.word_buffer else ""

    def _trigger_async(self):
        """워커 스레드를 띄워 Gemini 호출을 시작한다. 동시 호출은 1건만 허용."""
        with self._inflight_lock:
            if self._inflight or not self.word_buffer:
                return
            words = self.word_buffer.copy()
            self.word_buffer.clear()
            self._inflight = True

        threading.Thread(
            target=self._generate_worker,
            args=(words,),
            daemon=True,
        ).start()

    def _generate_worker(self, words):
        """워커 스레드 본체. Gemini 호출 결과(또는 오프라인 fallback)를 큐에 푸시한다."""
        if self._offline:
            sentence = " ".join(words)
        else:
            prompt = f"수화 단어: [{', '.join(words)}]"
            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=self._gen_config,
                )
                sentence = response.text.strip() if response.text else " ".join(words)
            except Exception as e:
                print(f"[SentenceBuilder] Gemini API error: {e}")
                sentence = " ".join(words)

        with self._inflight_lock:
            self._inflight = False
        self._completed.put(sentence)
