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
    GEMINI_SYSTEM_PROMPT, GEMINI_PERSONAS_PROMPT, TRIGGER_WORD, SILENCE_TRIGGER_SEC,
    SENTENCE_PERSONAS, SENTENCE_PERSONA, KSL_LABELS_EN,
)


def _english_words(words):
    """한국어 단어 목록 → LCD 표시 가능한 영어 라벨 나열."""
    return " ".join(KSL_LABELS_EN.get(w, "?") for w in words)


class SentenceBuilder:
    def __init__(self):
        # GEMINI_API_KEY가 비어있으면 오프라인 모드로 동작. 단어들을
        # 공백으로 이어 붙여 fallback 문장을 만든다. API 키가 있으면
        # 정상 LLM 경로.
        self.persona = SENTENCE_PERSONA
        if GEMINI_API_KEY:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._offline = False
            self._rebuild_config()                       # 단일 페르소나(reroll)용
            self._personas_config = self._build_personas_config()  # 전 페르소나 동시(옵션 B)
        else:
            print("[SentenceBuilder] No GEMINI_API_KEY in .env. "
                  "OFFLINE MODE: sentences = space-joined word list.")
            self._client = None
            self._gen_config = None
            self._personas_config = None
            self._offline = True

        self.word_buffer = []
        self._last_word_time = time.time()
        # 현재 단어셋의 문체별 캐시 {페르소나: (한국어, 영어)} — 한 번 생성해 재사용.
        self._persona_cache = {}
        self._cache_words = None  # 캐시가 대응하는 단어 목록(새 단어 들어오면 무효화)
        self._completed = queue.Queue()
        self._inflight = False
        self._inflight_lock = threading.Lock()

    def _build_personas_config(self):
        """전 페르소나를 한 번에 생성하는 config (GEMINI_PERSONAS_PROMPT)."""
        return types.GenerateContentConfig(
            system_instruction=GEMINI_PERSONAS_PROMPT,
            max_output_tokens=GEMINI_MAX_TOKENS * 2,  # 페르소나 수만큼 줄이 늘어남
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def set_persona(self, persona):
        """문장 문체 페르소나를 변경한다 (정중/친근).

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

        _trigger_async가 워커 시작 시점의 self._gen_config를 스냅샷해
        인자로 넘기므로, 생성 중에 페르소나가 바뀌어도 진행 중인 한 건은
        시작 시점 config로 결정적으로 완료된다.
        """
        self._gen_config = types.GenerateContentConfig(
            system_instruction=(
                GEMINI_SYSTEM_PROMPT + "\n" + SENTENCE_PERSONAS[self.persona]
            ),
            max_output_tokens=GEMINI_MAX_TOKENS,
            temperature=0.3,
            # gemini-2.5-flash는 thinking 모델 — 사고 토큰이 출력 예산을 잠식해
            # 문장이 잘린다. 짧은 번역 문장 생성엔 thinking 불필요하므로 끈다
            # (출력 예산 확보 + 지연 감소).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
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

        self._invalidate_cache()  # 새 단어 입력 = 이전 문장 캐시 무효
        self.word_buffer.append(word)
        self._last_word_time = time.time()

    def _invalidate_cache(self):
        self._persona_cache = {}
        self._cache_words = None

    def trigger_sentence(self):
        """버퍼 단어로 즉시 비동기 문장 생성을 시작한다 (물리 버튼·키 입력용).

        버퍼가 비어 있거나 이미 생성 중이면 시작하지 않고 False를 반환한다.
        """
        with self._inflight_lock:
            can_start = bool(self.word_buffer) and not self._inflight
        if can_start:
            self._trigger_async()
        return can_start

    def undo_last_word(self):
        """버퍼의 마지막 단어 1개를 제거한다 (오인식 복구, 단어 단위 초기화).

        제거한 단어를 반환하고, 버퍼가 비어 있으면 None을 반환한다.
        버퍼 조작은 모두 메인 스레드(인식 루프)에서 일어나므로 별도 락이 불필요하다.
        """
        if self.word_buffer:
            return self.word_buffer.pop()
        return None

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
        완료된 문장이 있으면 (한국어, 영어) 튜플 반환, 없으면 None.
        한국어는 TTS 음성용, 영어는 LCD·GUI 표시용 (한글 렌더링 불가 매체).
        메인 루프가 매 프레임 호출한다.
        """
        try:
            return self._completed.get_nowait()
        except queue.Empty:
            return None

    def get_buffer_preview(self, english=False):
        """실시간 표시용 현재 버퍼 내용. english=True면 LCD용 영어 라벨."""
        if not self.word_buffer:
            return ""
        if english:
            return " | ".join(KSL_LABELS_EN.get(w, "?") for w in self.word_buffer)
        return " | ".join(self.word_buffer)

    def _trigger_async(self):
        """전 페르소나 문장을 한 번의 호출로 생성한다(옵션 B). 동시 1건만 허용."""
        with self._inflight_lock:
            if self._inflight or not self.word_buffer:
                return
            words = self.word_buffer.copy()
            self.word_buffer.clear()
            self._inflight = True
        threading.Thread(
            target=self._generate_all_worker, args=(words,), daemon=True
        ).start()

    def replay_current(self):
        """현재 페르소나의 캐시 문장을 재출력한다 — Gemini 호출 없음.

        완료 재누름·페르소나 전환 후 완료에 사용. 캐시가 있으면 True.
        """
        result = self._persona_cache.get(self.persona)
        if result is None:
            return False
        self._completed.put(result)
        return True

    def reroll_current(self):
        """현재 페르소나 문장만 다시 생성한다(의미 오류 복구, undo 재누름).

        같은 단어로 해당 페르소나만 1회 재호출해 그 캐시만 갱신한다(다른 페르소나
        캐시는 유지). 캐시 단어가 있고 생성 중이 아니면 시작하고 True.
        """
        with self._inflight_lock:
            if self._inflight or not self._cache_words:
                return False
            words = list(self._cache_words)
            persona = self.persona
            gen_config = None if self._offline else self._gen_config
            self._inflight = True
        threading.Thread(
            target=self._reroll_worker, args=(words, gen_config, persona), daemon=True
        ).start()
        return True

    def _parse_personas(self, text, words):
        """'페르소나|한국어|영어' 줄들을 {페르소나:(ko,en)}로 파싱. 누락분은 fallback."""
        fallback = (" ".join(words), _english_words(words))
        out = {}
        for line in (text or "").splitlines():
            line = line.strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[0] in SENTENCE_PERSONAS:
                out[parts[0]] = (parts[1], parts[2])
        for p in SENTENCE_PERSONAS:
            out.setdefault(p, fallback)
        return out

    def _generate_all_worker(self, words):
        """전 페르소나를 한 번에 생성해 캐시를 채우고 현재 페르소나를 출력한다.

        inflight는 어떤 경로로 끝나든 finally에서 해제 — 1회 실패가 이후 트리거를
        영구 차단하지 않게 한다.
        """
        fallback = (" ".join(words), _english_words(words))
        try:
            if self._offline:
                cache = {p: fallback for p in SENTENCE_PERSONAS}
            else:
                prompt = f"수화 단어: [{', '.join(words)}]"
                try:
                    response = self._client.models.generate_content(
                        model=GEMINI_MODEL, contents=prompt,
                        config=self._personas_config,
                    )
                    cache = self._parse_personas(response.text, words)
                except Exception as e:
                    print(f"[SentenceBuilder] Gemini API error: {e}")
                    cache = {p: fallback for p in SENTENCE_PERSONAS}
            self._persona_cache = cache
            self._cache_words = list(words)
            self._completed.put(cache.get(self.persona, fallback))
        finally:
            with self._inflight_lock:
                self._inflight = False

    def _reroll_worker(self, words, gen_config, persona):
        """현재 페르소나 단일 재생성 — 두 줄(한국어/영어) 응답으로 해당 캐시만 갱신."""
        fallback = (" ".join(words), _english_words(words))
        try:
            if self._offline:
                result = fallback
            else:
                prompt = f"수화 단어: [{', '.join(words)}]"
                try:
                    response = self._client.models.generate_content(
                        model=GEMINI_MODEL, contents=prompt, config=gen_config,
                    )
                    lines = [l.strip() for l in (response.text or "").splitlines()
                             if l.strip()]
                    if lines:
                        english = lines[1] if len(lines) > 1 else _english_words(words)
                        result = (lines[0], english)
                    else:
                        result = fallback
                except Exception as e:
                    print(f"[SentenceBuilder] Gemini API error: {e}")
                    result = fallback
            self._persona_cache[persona] = result  # 해당 페르소나 캐시만 갱신
            self._completed.put(result)
        finally:
            with self._inflight_lock:
                self._inflight = False
