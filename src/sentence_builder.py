"""
sentence_builder.py
인식된 수화 단어를 버퍼에 축적하고,
트리거 조건 충족 시 Gemini API를 호출해 자연어 문장을 생성합니다.
"""

import time
import google.generativeai as genai

from config.settings import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS,
    GEMINI_SYSTEM_PROMPT, TRIGGER_WORD, SILENCE_TRIGGER_SEC
)


class SentenceBuilder:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=GEMINI_SYSTEM_PROMPT
        )

        self.word_buffer = []
        self._last_word_time = time.time()

    def add_word(self, word):
        """
        인식된 단어를 버퍼에 추가합니다.
        TRIGGER_WORD("완료")가 들어오면 즉시 문장 생성을 트리거합니다.
        """
        if word == TRIGGER_WORD:
            return self._generate_sentence()

        self.word_buffer.append(word)
        self._last_word_time = time.time()
        return None

    def check_silence_trigger(self):
        """
        마지막 단어 인식 후 SILENCE_TRIGGER_SEC 이상 경과하면
        자동으로 문장 생성을 트리거합니다.
        """
        if not self.word_buffer:
            return None
        if time.time() - self._last_word_time >= SILENCE_TRIGGER_SEC:
            return self._generate_sentence()
        return None

    def _generate_sentence(self):
        """
        Gemini API를 호출해 자연어 문장을 생성합니다.

        Returns:
            str — 생성된 한국어 문장
            str — 오프라인 fallback 단어 나열 문자열
        """
        if not self.word_buffer:
            return None

        words = self.word_buffer.copy()
        self.word_buffer.clear()

        prompt = f"수화 단어: [{', '.join(words)}]"

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.3
                )
            )
            return response.text.strip()

        except Exception as e:
            print(f"[SentenceBuilder] Gemini API error: {e}")
            # 오프라인 fallback: 단어 나열
            return " ".join(words)

    def get_buffer_preview(self):
        """LCD 실시간 표시용 현재 버퍼 내용."""
        return " | ".join(self.word_buffer) if self.word_buffer else ""
