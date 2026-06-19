"""
test_settings_override.py
config/settings.py 튜닝 값의 기본값·환경변수 오버라이드 검증 (CLAUDE.md §2.2).

settings는 모듈 임포트 시점에 환경변수를 읽으므로, os.environ을 조작한 뒤
importlib.reload로 재평가한다. load_dotenv()는 이미 설정된 환경변수를
덮어쓰지 않으므로 os.environ 주입이 .env 파일보다 우선한다.
"""

import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.settings as settings


def _reload_settings():
    return importlib.reload(settings)


def test_duplicate_filter_default():
    """키 부재 시 기본 3.0초 — 동작 유지 시 1.5초마다 같은 단어가 버퍼에
    쌓이던 실측(2026-06-12 리허설, "help help help") 대응. 침묵 트리거
    복원값(3.0)과 정렬해 한 동작당 한 단어가 되도록 한다."""
    os.environ.pop("DUPLICATE_FILTER_SEC", None)
    s = _reload_settings()
    assert s.DUPLICATE_FILTER_SEC == 3.0, s.DUPLICATE_FILTER_SEC
    print("[PASS] test_duplicate_filter_default")


def test_duplicate_filter_env_override():
    """환경변수로 오버라이드 가능해야 한다 (하드코딩 금지 — CLAUDE.md §2.2)."""
    os.environ["DUPLICATE_FILTER_SEC"] = "2.0"
    try:
        s = _reload_settings()
        assert s.DUPLICATE_FILTER_SEC == 2.0, s.DUPLICATE_FILTER_SEC
    finally:
        os.environ.pop("DUPLICATE_FILTER_SEC", None)
        _reload_settings()
    print("[PASS] test_duplicate_filter_env_override")


if __name__ == "__main__":
    test_duplicate_filter_default()
    test_duplicate_filter_env_override()
    print("\nAll tests done.")
