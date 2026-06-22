"""
test_data_split.py
held-out 시연자 분리(model/data_split.py) 검증 — PR #9 리뷰 #6.

판정 지표("held-out 시연자 정확도")의 전제:
  - test = held-out 시연자의 *원본*만
  - held-out 시연자의 증강본은 train·test 양쪽에서 제외 (누출 차단)
  - "local" 그룹(collect_data.py 산출물)은 항상 train (데모 화자 포함)
TF 무의존 — CI 실행 가능.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.data_split import parse_group, is_augmented, split_holdout


# ── parse_group ───────────────────────────────────────────────

def test_parse_group_real_pattern():
    """AI Hub 변환 파일명에서 시연자 ID를 추출한다."""
    assert parse_group("NIA_SL_WORD1157_REAL01_F__w00.csv") == "REAL01"
    assert parse_group("NIA_SL_WORD0742_REAL16_D__w03.csv") == "REAL16"
    print("[PASS] test_parse_group_real_pattern")


def test_parse_group_augmented_inherits():
    """증강본 파일명도 원본 stem을 포함하므로 같은 그룹으로 귀속된다."""
    assert parse_group("NIA_SL_WORD1157_REAL01_F__w00__aug2.csv") == "REAL01"
    print("[PASS] test_parse_group_augmented_inherits")


def test_parse_group_local_fallback():
    """REAL 패턴이 없는 파일(직접 수집·구버전 변환)은 'local' 그룹."""
    assert parse_group("0001.csv") == "local"
    assert parse_group("aihub_0042.csv") == "local"
    assert parse_group("data/landmarks/나/0001.csv") == "local"
    print("[PASS] test_parse_group_local_fallback")


def test_is_augmented():
    assert is_augmented("NIA_SL_WORD1157_REAL01_F__w00__aug2.csv")
    assert not is_augmented("NIA_SL_WORD1157_REAL01_F__w00.csv")
    assert not is_augmented("0001.csv")
    print("[PASS] test_is_augmented")


# ── split_holdout ─────────────────────────────────────────────

def _synthetic_files(real_ids=("REAL01", "REAL02", "REAL03", "REAL04"),
                     n_orig=3, factor=2, n_local=2):
    """시연자별 원본 n_orig개 + 증강 factor배 + local 파일을 합성한다."""
    files = []
    for rid in real_ids:
        for i in range(n_orig):
            stem = f"NIA_SL_WORD1157_{rid}_F__w{i:02d}"
            files.append(f"data/landmarks/나/{stem}.csv")
            for k in range(factor):
                files.append(f"data/augmented/나/{stem}__aug{k}.csv")
    for i in range(n_local):
        files.append(f"data/landmarks/나/{i:04d}.csv")
    return files


def test_split_holdout_excludes_holdout_from_train():
    """held-out 시연자의 원본·증강본이 train에 0건이어야 한다."""
    files = _synthetic_files()
    train_mask, test_mask, holdout = split_holdout(files, n_holdout=1, seed=42)
    assert len(holdout) == 1
    groups = np.array([parse_group(f) for f in files])
    assert not np.any(np.isin(groups[train_mask], holdout)), \
        "held-out 시연자 샘플이 train에 포함됨"
    print("[PASS] test_split_holdout_excludes_holdout_from_train")


def test_split_holdout_test_is_originals_only():
    """test는 held-out 시연자의 원본만 — 증강본은 양쪽 모두 제외."""
    files = _synthetic_files()
    train_mask, test_mask, holdout = split_holdout(files, n_holdout=1, seed=42)
    files_arr = np.array(files)
    test_files = files_arr[test_mask]
    assert len(test_files) > 0
    assert all(parse_group(f) in holdout for f in test_files)
    assert not any(is_augmented(f) for f in test_files), "증강본이 test에 포함됨"
    # held-out 증강본은 train에도 test에도 없어야 한다
    dropped = [f for f in files
               if parse_group(f) in holdout and is_augmented(f)]
    in_train = set(files_arr[train_mask])
    in_test = set(test_files)
    assert all(f not in in_train and f not in in_test for f in dropped)
    print("[PASS] test_split_holdout_test_is_originals_only")


def test_split_holdout_local_always_train():
    """local 그룹은 항상 train — held-out 후보에서 제외된다."""
    files = _synthetic_files()
    train_mask, test_mask, holdout = split_holdout(files, n_holdout=1, seed=42)
    assert "local" not in holdout
    files_arr = np.array(files)
    local_files = {f for f in files if parse_group(f) == "local"}
    assert local_files.issubset(set(files_arr[train_mask]))
    print("[PASS] test_split_holdout_local_always_train")


def test_split_holdout_deterministic():
    """같은 seed면 같은 held-out 시연자가 선택된다 (train/evaluate 재현성)."""
    files = _synthetic_files()
    _, _, h1 = split_holdout(files, n_holdout=2, seed=42)
    _, _, h2 = split_holdout(files, n_holdout=2, seed=42)
    assert list(h1) == list(h2)
    print("[PASS] test_split_holdout_deterministic")


def test_split_holdout_fallback_when_few_groups():
    """REAL 그룹 < 3이면 (None, None, []) — 호출자가 랜덤 분할로 폴백.
    smoke test·KSL_LABELS_OVERRIDE 등 소규모 데이터 경로 보존."""
    files = _synthetic_files(real_ids=("REAL01",))
    train_mask, test_mask, holdout = split_holdout(files, n_holdout=1, seed=42)
    assert train_mask is None and test_mask is None and len(holdout) == 0
    print("[PASS] test_split_holdout_fallback_when_few_groups")


if __name__ == "__main__":
    test_parse_group_real_pattern()
    test_parse_group_augmented_inherits()
    test_parse_group_local_fallback()
    test_is_augmented()
    test_split_holdout_excludes_holdout_from_train()
    test_split_holdout_test_is_originals_only()
    test_split_holdout_local_always_train()
    test_split_holdout_deterministic()
    test_split_holdout_fallback_when_few_groups()
    print("\nAll tests done.")
