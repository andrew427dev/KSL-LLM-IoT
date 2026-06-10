"""
test_smoke.py
경량 정적 검증 — mediapipe / tensorflow / tflite-runtime 없이 실행 가능.

CI에서 매 PR/push마다 돌려:
  - 설정 상수(FEATURE_DIM / 슬롯 인덱스 / INPUT_SHAPE) 일관성
  - feature_format의 scale 정규화·presence flag·wrist_vec 수치 회귀
  - augment.flip_horizontal의 거울 효과 수치 회귀 (131 레이아웃)
  - hand_tracker.py 소스에 핵심 분기(score 임계값·라벨 충돌·x fallback) 존재
  - train.py에 LabelEncoder 부재 (정렬 인코딩 버그 회귀 가드)
  - 핵심 모듈에 단일 손 차원(63) 하드코딩 잔재 부재

를 검증한다. mediapipe/tflite 휠을 받지 않으므로 RPi/CI 모두 빠르다.

로컬 실행:
    python tests/test_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config.settings import (
    INPUT_SHAPE, NUM_HANDS, NUM_LANDMARKS, NUM_AXES, SEQUENCE_LENGTH,
    FEATURE_DIM, LEFT_SLOT_START, RIGHT_SLOT_START,
    WRIST_VEC_START, PRESENCE_FLAG_START,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_settings_consistency():
    """131차원 레이아웃 상수들의 상호 일관성."""
    assert RIGHT_SLOT_START == NUM_LANDMARKS * NUM_AXES == 63
    assert WRIST_VEC_START == NUM_HANDS * NUM_LANDMARKS * NUM_AXES == 126
    assert PRESENCE_FLAG_START == WRIST_VEC_START + 3 == 129
    assert FEATURE_DIM == PRESENCE_FLAG_START + 2 == 131
    expected = (SEQUENCE_LENGTH, FEATURE_DIM)
    assert INPUT_SHAPE == expected, f"INPUT_SHAPE={INPUT_SHAPE} != {expected}"
    assert INPUT_SHAPE == (30, 131), f"INPUT_SHAPE={INPUT_SHAPE}, expected (30, 131)"
    print(f"[OK] settings: INPUT_SHAPE={INPUT_SHAPE}, FEATURE_DIM={FEATURE_DIM}")


def _fake_hand(wrist, scale=0.1):
    """손목 wrist, landmark9가 손목에서 scale 거리인 (21,3) 절대좌표."""
    coords = np.tile(np.asarray(wrist, dtype=np.float32), (21, 1))
    for i in range(21):
        coords[i, 0] += (i % 7) * (scale / 7)
        coords[i, 1] += (i % 5) * (scale / 5)
    coords[9] = coords[0] + [scale, 0.0, 0.0]
    return coords


def test_feature_format():
    """normalize_hand·build_feature_vector의 수치 회귀 — numpy만 사용."""
    from src.feature_format import normalize_hand, build_feature_vector

    # scale 정규화: landmark9의 손목 상대 거리가 1
    norm, scale = normalize_hand(_fake_hand((0.3, 0.5, 0.0), scale=0.25))
    assert abs(scale - 0.25) < 1e-6
    lm9 = norm[9 * 3:9 * 3 + 3]
    assert abs(np.linalg.norm(lm9) - 1.0) < 1e-5

    # 퇴화 케이스: 모든 점이 한 곳 → (None, None)
    collapsed = np.tile(np.float32([0.5, 0.5, 0.0]), (21, 1))
    assert normalize_hand(collapsed) == (None, None)

    # 양손: wrist_vec = (우손목-좌손목)/scale 평균, presence 둘 다 1
    left = _fake_hand((0.3, 0.5, 0.0), scale=0.1)
    right = _fake_hand((0.7, 0.5, 0.0), scale=0.1)
    vec = build_feature_vector(left, right)
    assert vec.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(
        vec[WRIST_VEC_START:WRIST_VEC_START + 3], [4.0, 0.0, 0.0], atol=1e-4
    )
    assert vec[PRESENCE_FLAG_START] == 1.0 and vec[PRESENCE_FLAG_START + 1] == 1.0

    # 한 손: 반대 슬롯 zero, wrist_vec zero, flag (1,0)
    vec_l = build_feature_vector(left, None)
    assert np.all(vec_l[RIGHT_SLOT_START:WRIST_VEC_START] == 0.0)
    assert np.all(vec_l[WRIST_VEC_START:WRIST_VEC_START + 3] == 0.0)
    assert vec_l[PRESENCE_FLAG_START] == 1.0 and vec_l[PRESENCE_FLAG_START + 1] == 0.0

    # 양손 부재 → None
    assert build_feature_vector(None, None) is None
    print("[OK] feature_format: scale 정규화 + wrist_vec + presence flag")


def test_augment_flip_horizontal():
    """거울 효과 = 손내부 x 반전 + wrist_vec y·z 반전 + 슬롯/flag swap."""
    from model.augment import flip_horizontal

    seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    seq[:, LEFT_SLOT_START:RIGHT_SLOT_START] = 1.0   # LEFT (x=y=z=1)
    seq[:, RIGHT_SLOT_START:WRIST_VEC_START] = 2.0   # RIGHT (x=y=z=2)
    seq[:, WRIST_VEC_START:WRIST_VEC_START + 3] = [3.0, 4.0, 5.0]
    seq[:, PRESENCE_FLAG_START] = 1.0                # 좌손만 존재로 가정
    flipped = flip_horizontal(seq)

    # 결과 LEFT 슬롯에는 원래 RIGHT(2.0)가 들어가되 x는 부호 반전 (-2.0).
    assert flipped[0, LEFT_SLOT_START] == -2.0
    assert flipped[0, LEFT_SLOT_START + 1] == 2.0  # y unchanged
    # 결과 RIGHT 슬롯에는 원래 LEFT(1.0)가 들어가되 x는 부호 반전 (-1.0).
    assert flipped[0, RIGHT_SLOT_START] == -1.0
    assert flipped[0, RIGHT_SLOT_START + 1] == 1.0
    # wrist_vec: x 불변, y·z 부호 반전
    np.testing.assert_allclose(
        flipped[0, WRIST_VEC_START:WRIST_VEC_START + 3], [3.0, -4.0, -5.0]
    )
    # presence flag swap: (1,0) → (0,1)
    assert flipped[0, PRESENCE_FLAG_START] == 0.0
    assert flipped[0, PRESENCE_FLAG_START + 1] == 1.0
    print("[OK] augment.flip_horizontal: x-flip + wrist_vec y/z-flip + slot/flag swap")


def test_hand_tracker_source_markers():
    """hand_tracker.py 소스에 양손/충돌/fallback 처리 핵심 분기가 존재."""
    path = os.path.join(_REPO_ROOT, "src", "hand_tracker.py")
    src = open(path, encoding="utf-8").read()
    required = [
        "max_num_hands=NUM_HANDS",
        "HANDEDNESS_SCORE_THRESHOLD",
        "multi_handedness",
        "build_feature_vector",     # feature_format 공용 모듈 사용
        "handedness conflict",      # 충돌 warning 메시지
        "unlabeled.sort",           # x좌표 fallback
    ]
    for needle in required:
        assert needle in src, f"hand_tracker.py missing marker: {needle!r}"
    print("[OK] hand_tracker.py: 양손 + score 임계값 + 충돌/fallback 분기 존재")


def test_sentence_persona_settings():
    """페르소나 설정 일관성 — 기본값이 등록된 페르소나여야 한다."""
    from config.settings import (
        SENTENCE_PERSONAS, SENTENCE_PERSONA, KSL_LABELS, TRIGGER_WORD,
    )
    assert set(SENTENCE_PERSONAS) == {"정중", "친근", "간단"}
    assert SENTENCE_PERSONA in SENTENCE_PERSONAS
    assert all(v.strip() for v in SENTENCE_PERSONAS.values())
    # KSL_LABELS 불변식도 함께 검증
    assert len(KSL_LABELS) == len(set(KSL_LABELS)) == 30
    assert KSL_LABELS[-1] == TRIGGER_WORD
    print(f"[OK] persona settings: {list(SENTENCE_PERSONAS)}, "
          f"default={SENTENCE_PERSONA}, labels=30")


def test_english_label_mapping():
    """LCD·GUI 표시용 영어 라벨 — 30개 라벨 전부 매핑되고 ASCII여야 한다.
    (HD44780 LCD·cv2.putText는 한글 렌더링 불가)"""
    from config.settings import KSL_LABELS, KSL_LABELS_EN
    missing = [w for w in KSL_LABELS if w not in KSL_LABELS_EN]
    assert not missing, f"영어 라벨 누락: {missing}"
    non_ascii = {w: en for w, en in KSL_LABELS_EN.items()
                 if not en.isascii() or not en.strip()}
    assert not non_ascii, f"비ASCII/빈 영어 라벨: {non_ascii}"
    print(f"[OK] KSL_LABELS_EN: {len(KSL_LABELS_EN)}개 전부 ASCII 매핑")


def test_train_has_no_label_encoder():
    """train.py에 sklearn LabelEncoder 사용 금지 — 유니코드 정렬로 학습 인덱스가
    추론(KSL_LABELS[label_idx])과 어긋나는 버그 회귀 가드."""
    src = open(os.path.join(_REPO_ROOT, "model", "train.py"), encoding="utf-8").read()
    # 실제 사용 패턴만 탐지 (docstring의 경고 문구는 허용)
    for usage in ("LabelEncoder(", "import LabelEncoder", "LabelEncoder,"):
        assert usage not in src, (
            f"train.py에서 {usage!r} 발견 — KSL_LABELS 원본 순서 인코딩(LABEL_TO_IDX)을 사용할 것"
        )
    assert "LABEL_TO_IDX" in src
    print("[OK] train.py: LabelEncoder 부재, 원본 순서 인코딩 사용")


def test_no_single_hand_hardcoding():
    """핵심 모듈에 (30, 63) / np.zeros(63) 등 단일 손 차원 하드코딩이 없어야 한다."""
    bad_patterns = [
        "shape == (30, 63)",
        "shape == (SEQUENCE_LENGTH, 63)",
        "shape == (30, 126)",
        "shape == (SEQUENCE_LENGTH, 126)",
        "np.zeros(63",
        "np.zeros(126",
        "np.random.rand(63)",
        "np.random.rand(126)",
    ]
    files = [
        "model/train.py",
        "model/augment.py",
        "tests/test_classifier.py",
        "src/classifier.py",
        "src/hand_tracker.py",
        "src/feature_format.py",
    ]
    for rel in files:
        s = open(os.path.join(_REPO_ROOT, rel), encoding="utf-8").read()
        for p in bad_patterns:
            assert p not in s, f"{rel}: 고정 차원 잔재 발견 {p!r}"
    print("[OK] no hardcoded hand-dimension in core modules")


if __name__ == "__main__":
    test_settings_consistency()
    test_feature_format()
    test_augment_flip_horizontal()
    test_hand_tracker_source_markers()
    test_sentence_persona_settings()
    test_english_label_mapping()
    test_train_has_no_label_encoder()
    test_no_single_hand_hardcoding()
    print("\n=== smoke test PASS ===")
