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
    GEMINI_SYSTEM_PROMPT, TRIGGER_WORD, SILENCE_TRIGGER_SEC
)


class SentenceBuilder:
    def __init__(self):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._gen_config = types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_PROMPT,
            max_output_tokens=GEMINI_MAX_TOKENS,
            temperature=0.3,
        )

        self.word_buffer = []
        self._last_word_time = time.time()
        self._completed = queue.Queue()
        self._inflight = False
        self._inflight_lock = threading.Lock()

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

    def check_silence_trigger(self):
        """
        마지막 단어 인식 후 SILENCE_TRIGGER_SEC 이상 경과하면
        비동기 문장 생성을 시작한다.
        """
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
        """워커 스레드 본체. Gemini 호출 결과를 큐에 푸시한다."""
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
        finally:
            with self._inflight_lock:
                self._inflight = False

        self._completed.put(sentence)
