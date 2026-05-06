"""
tts_output.py
생성된 문장을 음성으로 출력합니다.
온라인: gTTS (Google TTS)
오프라인 fallback: pyttsx3
"""

import os
import threading
from config.settings import TTS_LANGUAGE, TTS_OFFLINE_MODE


class TTSOutput:
    def __init__(self):
        self._lock = threading.Lock()

        if TTS_OFFLINE_MODE:
            self._init_offline()
        else:
            self._engine_type = "gtts"

    def _init_offline(self):
        import pyttsx3
        self._offline_engine = pyttsx3.init()
        self._offline_engine.setProperty('rate', 150)
        self._engine_type = "pyttsx3"

    def speak(self, text):
        """비동기로 음성 출력 (인식 루프 블로킹 방지)."""
        thread = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
        thread.start()

    def _speak_sync(self, text):
        with self._lock:
            if self._engine_type == "gtts":
                self._speak_gtts(text)
            else:
                self._speak_pyttsx3(text)

    def _speak_gtts(self, text):
        try:
            from gtts import gTTS
            import pygame

            tts = gTTS(text=text, lang=TTS_LANGUAGE)
            tts.save("/tmp/ksl_output.mp3")

            pygame.mixer.init()
            pygame.mixer.music.load("/tmp/ksl_output.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except Exception as e:
            print(f"[TTS] gTTS error: {e}, falling back to pyttsx3")
            self._init_offline()
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text):
        self._offline_engine.say(text)
        self._offline_engine.runAndWait()
