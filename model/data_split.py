"""
data_split.py
held-out 시연자 단위 train/test 분리 — PR #9 리뷰 #6.

배경: 슬라이딩 윈도우(같은 영상의 인접 윈도우)와 증강본(원본의 변형)이
샘플 단위 랜덤 분할에서 train/test 양쪽에 들어가는 구조적 누출이 있었다
— Test Accuracy가 1.0000으로 포화되어 일반화 측정이 불가능했다.
시연자(REAL01~) 단위로 분리하면 held-out 시연자의 어떤 변형도 학습에
노출되지 않아, HANDOFF §1.4의 판정 지표("held-out 시연자 정확도" — FPS
무관한 순수 일반화 측정)가 된다.

파일명 규약 (그룹 식별의 근거):
  - convert_aihub.py: {키포인트 디렉터리명}__w{윈도우}.csv
      예: NIA_SL_WORD1157_REAL01_F__w00.csv → 그룹 REAL01
  - model/augment.py: {원본 stem}__aug{k}.csv → 원본 그룹 상속
  - collect_data.py 등 REAL 패턴이 없는 파일 → 그룹 "local",
      항상 train (HANDOFF §4.2 — 데모 화자 데이터는 학습에 포함)

numpy 외 의존성 없음 (TF 무의존 — CI 실행 가능).
"""

import os
import re
import numpy as np

LOCAL_GROUP = "local"

# held-out 후보가 이보다 적으면 시연자 분리가 무의미 — 호출자는 랜덤 분할 폴백
MIN_REAL_GROUPS = 3

_REAL_PATTERN = re.compile(r"_(REAL\d+)_")
_AUG_PATTERN = re.compile(r"__aug\d+\.csv$")


def parse_group(fname):
    """파일명(경로 허용)에서 시연자 그룹을 추출한다.

    Returns: "REAL01" 형태 시연자 ID, 패턴 부재 시 LOCAL_GROUP.
    """
    m = _REAL_PATTERN.search(os.path.basename(fname))
    return m.group(1) if m else LOCAL_GROUP


def is_augmented(fname):
    """model/augment.py 산출물({stem}__aug{k}.csv) 여부."""
    return bool(_AUG_PATTERN.search(os.path.basename(fname)))


def split_holdout(files, n_holdout=2, seed=42):
    """시연자 단위 held-out 분리 마스크를 만든다.

    Args:
        files: 샘플 파일 경로 리스트 (load_dataset 순서와 일치해야 함).
        n_holdout: held-out 시연자 수.
        seed: held-out 선택 재현용 시드.
    Returns:
        (train_mask, test_mask, holdout_groups)
        - train_mask/test_mask: bool ndarray (len(files),).
          held-out 시연자의 증강본은 양쪽 마스크 모두 False (누출 차단).
        - holdout_groups: 선택된 시연자 ID 리스트.
        REAL 그룹이 MIN_REAL_GROUPS 미만이면 (None, None, []) —
        호출자는 기존 샘플 단위 랜덤 분할로 폴백한다.
    """
    groups = np.array([parse_group(f) for f in files])
    augmented = np.array([is_augmented(f) for f in files])

    real_groups = sorted(g for g in set(groups) if g != LOCAL_GROUP)
    if len(real_groups) < MIN_REAL_GROUPS:
        return None, None, []

    n_holdout = min(n_holdout, len(real_groups) - 1)
    rng = np.random.RandomState(seed)
    holdout = sorted(rng.choice(real_groups, size=n_holdout, replace=False))

    in_holdout = np.isin(groups, holdout)
    train_mask = ~in_holdout
    test_mask = in_holdout & ~augmented
    return train_mask, test_mask, list(holdout)
