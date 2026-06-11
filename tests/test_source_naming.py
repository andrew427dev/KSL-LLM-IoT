"""
test_source_naming.py
변환·증강 산출물 파일명이 시연자 그룹 정보를 보존하는지 검증 — PR #9 리뷰 #6.

held-out 분리(model/data_split.py)는 파일명 규약에 의존한다:
  convert_aihub: {키포인트 디렉터리명}__w{윈도우}.csv
  augment:       {원본 stem}__aug{k}.csv
이 규약이 깨지면 모든 파일이 "local" 그룹이 되어 분리가 무음 폴백된다.
TF 무의존 — CI 실행 가능.
"""
import os
import sys
import tempfile
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import KSL_LABELS, SEQUENCE_LENGTH, FEATURE_DIM, PRESENCE_FLAG_START
from model.data_split import parse_group
from convert_aihub import window_filename
import model.augment as augment


def test_convert_window_filename_preserves_source():
    """변환 파일명 = {디렉터리명}__w{NN}.csv — 시연자·각도 보존."""
    fname = window_filename("NIA_SL_WORD1157_REAL01_F", 3)
    assert fname == "NIA_SL_WORD1157_REAL01_F__w03.csv"
    assert parse_group(fname) == "REAL01"
    print("[PASS] test_convert_window_filename_preserves_source")


def test_augment_filename_inherits_original_stem():
    """증강 파일명 = {원본 stem}__aug{k}.csv — 원본 그룹 상속."""
    label = KSL_LABELS[0]
    orig_name = "NIA_SL_WORD1157_REAL01_F__w00.csv"

    seq = np.random.uniform(-1, 1, (SEQUENCE_LENGTH, FEATURE_DIM)).astype(np.float32)
    seq[:, PRESENCE_FLAG_START:] = 1.0  # 양손 존재 — 증강 불변식 경로 단순화

    orig_dirs = augment.LANDMARKS_DIR, augment.AUGMENTED_DIR
    with tempfile.TemporaryDirectory() as tmp:
        augment.LANDMARKS_DIR = os.path.join(tmp, "landmarks")
        augment.AUGMENTED_DIR = os.path.join(tmp, "augmented")
        try:
            src_dir = os.path.join(augment.LANDMARKS_DIR, label)
            os.makedirs(src_dir)
            np.savetxt(os.path.join(src_dir, orig_name), seq, delimiter=",")

            n = augment.augment_label(label, factor=2)
            assert n == 2

            out_files = sorted(os.listdir(os.path.join(augment.AUGMENTED_DIR, label)))
            assert out_files == [
                "NIA_SL_WORD1157_REAL01_F__w00__aug0.csv",
                "NIA_SL_WORD1157_REAL01_F__w00__aug1.csv",
            ], f"증강 파일명이 원본 stem을 상속하지 않음: {out_files}"
            assert all(parse_group(f) == "REAL01" for f in out_files)
        finally:
            augment.LANDMARKS_DIR, augment.AUGMENTED_DIR = orig_dirs
    print("[PASS] test_augment_filename_inherits_original_stem")


if __name__ == "__main__":
    test_convert_window_filename_preserves_source()
    test_augment_filename_inherits_original_stem()
    print("\nAll tests done.")
