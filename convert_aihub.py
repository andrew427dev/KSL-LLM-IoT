"""
convert_aihub.py
AI Hub 수어 영상 데이터셋의 3D 키포인트 JSON을
프로젝트 CSV 포맷(30프레임 × 126차원)으로 변환한다.

변환 흐름:
  1. 형태소 JSON에서 WORD번호 → 한국어 단어 매핑 구축
  2. KSL_LABELS에 해당하는 WORD번호 필터링
  3. 키포인트 JSON에서 hand_left/right_keypoints_3d 추출
  4. 손목 기준 정규화 → [LEFT_63 | RIGHT_63] = 126차원
  5. 슬라이딩 윈도우로 30프레임 시퀀스 생성 → CSV 저장

Usage:
    python convert_aihub.py --dataset /path/to/aihub
    python convert_aihub.py --dataset /path/to/aihub --angles F D
    python convert_aihub.py --dataset /path/to/aihub --stride 10
    python convert_aihub.py --dataset /path/to/aihub --scan  # 매칭 현황만 출력
"""

import json
import os
import argparse
import glob
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.settings import KSL_LABELS, SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_AXES

PER_HAND_DIM = NUM_LANDMARKS * NUM_AXES  # 63
FRAME_DIM = PER_HAND_DIM * 2  # 126

# 양손 모두 zero인 프레임이 시퀀스의 절반 이상이면 해당 샘플은 폐기
MIN_NONZERO_RATIO = 0.5


# ── 1단계: WORD번호 → 한국어 단어 매핑 ────────────────────────

def build_word_mapping(dataset_root):
    """형태소 JSON에서 WORD번호 → 한국어 단어 매핑을 구축한다."""
    mapping = {}
    count = 0
    for root, _dirs, files in os.walk(dataset_root):
        for f in files:
            if not f.endswith("_morpheme.json"):
                continue
            parts = f.split("_")
            if len(parts) < 3 or not parts[2].startswith("WORD"):
                continue
            word_num = parts[2][4:]  # "WORD0001" → "0001"
            if word_num in mapping:
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, encoding="utf-8") as fp:
                    d = json.load(fp)
                for item in d.get("data", []):
                    for attr in item.get("attributes", []):
                        name = attr.get("name", "").strip()
                        if name:
                            mapping[word_num] = name
                            count += 1
                            break
                    if word_num in mapping:
                        break
            except (json.JSONDecodeError, KeyError, OSError):
                pass
    print(f"[Convert] 형태소 매핑 완료: {len(mapping)}개 단어")
    return mapping


# ── 2단계: KSL_LABELS 매칭 ────────────────────────────────────

def match_labels(word_mapping, labels):
    """KSL_LABELS와 매칭되는 WORD번호를 찾는다.
    정확 매칭 우선, 어근 매칭(한쪽이 다른 쪽에 포함) 보조.
    Returns: {label: [word_num, ...]}
    """
    # 역매핑: 한국어 단어 → WORD번호 리스트
    korean_to_nums = {}
    for num, word in word_mapping.items():
        korean_to_nums.setdefault(word, []).append(num)

    result = {}
    for label in labels:
        nums = []
        # 정확 매칭
        if label in korean_to_nums:
            nums.extend(korean_to_nums[label])
        # 어근 매칭: "감사합니다" ↔ "감사", "행복하다" ↔ "행복"
        # 양쪽 모두 2글자 이상이어야 오매칭 방지 ("나" ⊂ "낚시" 같은 케이스 차단)
        if len(label) >= 2:
            for korean_word, wnums in korean_to_nums.items():
                if korean_word == label:
                    continue
                if len(korean_word) >= 2 and (korean_word in label or label in korean_word):
                    nums.extend(wnums)
        if nums:
            result[label] = sorted(set(nums))
    return result


# ── 3단계: 키포인트 디렉터리 탐색 ─────────────────────────────

def find_keypoint_dirs(dataset_root):
    """키포인트 JSON을 포함하는 디렉터리를 검색한다.
    Returns: {dirname: abspath}
    """
    kp_dirs = {}
    for root, _dirs, files in os.walk(dataset_root):
        if any(f.endswith("_keypoints.json") for f in files[:5]):
            dirname = os.path.basename(root)
            kp_dirs[dirname] = root
    print(f"[Convert] 키포인트 디렉터리: {len(kp_dirs)}개 발견")
    return kp_dirs


# ── 4단계: 프레임 변환 핵심 로직 ──────────────────────────────

def extract_hand_3d(people, hand_key):
    """3D 키포인트에서 (x,y,z) 추출 후 손목 기준 정규화.
    AI Hub 포맷: 21 joints × 4 (x, y, z, confidence) = 84개 값.
    Returns: 63차원 벡터 (21 × 3). 데이터 없으면 zero 벡터.
    """
    kp = people.get(hand_key, [])
    if not kp or len(kp) < NUM_LANDMARKS * 4:
        return np.zeros(PER_HAND_DIM, dtype=np.float32)

    coords = np.array(
        [[kp[i * 4], kp[i * 4 + 1], kp[i * 4 + 2]] for i in range(NUM_LANDMARKS)],
        dtype=np.float32,
    )  # (21, 3)

    # 손목(joint 0) 기준 상대 좌표 — hand_tracker.py:_normalize_hand()와 동일 로직
    wrist = coords[0].copy()
    coords -= wrist

    return coords.flatten()


def load_sequence_frames(kp_dir):
    """한 시퀀스 디렉터리의 전체 프레임을 로드한다.
    Returns: (N, 126) ndarray 또는 None.
    """
    json_files = sorted(glob.glob(os.path.join(kp_dir, "*_keypoints.json")))
    if not json_files:
        return None

    frames = []
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue

        people = data.get("people", {})
        left = extract_hand_3d(people, "hand_left_keypoints_3d")
        right = extract_hand_3d(people, "hand_right_keypoints_3d")
        frames.append(np.concatenate([left, right]))

    if not frames:
        return None
    return np.array(frames, dtype=np.float32)


def make_samples(sequence, window_size, stride):
    """슬라이딩 윈도우로 (window_size, 126) 샘플들을 생성한다."""
    samples = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[start : start + window_size]
        nonzero = np.count_nonzero(window.sum(axis=1)) / window_size
        if nonzero >= MIN_NONZERO_RATIO:
            samples.append(window)
    return samples


# ── 메인 변환 ─────────────────────────────────────────────────

def scan_only(dataset_root, labels):
    """매칭 현황만 출력하고 종료한다."""
    word_mapping = build_word_mapping(dataset_root)
    label_matches = match_labels(word_mapping, labels)
    kp_dirs = find_keypoint_dirs(dataset_root)

    print(f"\n{'라벨':12s} | {'WORD번호':30s} | {'키포인트 시퀀스':10s}")
    print("-" * 62)
    for label in labels:
        nums = label_matches.get(label, [])
        if not nums:
            print(f"{label:12s} | {'(없음)':30s} | -")
            continue
        # 이 WORD번호에 해당하는 키포인트 디렉터리 개수
        kp_count = 0
        for num in nums:
            pattern = f"WORD{num}_"
            kp_count += sum(1 for d in kp_dirs if pattern in d)
        nums_str = ", ".join(f"WORD{n}" for n in nums[:5])
        if len(nums) > 5:
            nums_str += f" ... (+{len(nums)-5})"
        print(f"{label:12s} | {nums_str:30s} | {kp_count}개")

    total_kp = sum(
        sum(1 for d in kp_dirs if f"WORD{n}_" in d)
        for nums in label_matches.values()
        for n in nums
    )
    print(f"\n총 매칭 라벨: {len(label_matches)}/{len(labels)}, 키포인트 시퀀스: {total_kp}개")


def convert(dataset_root, output_dir, angles, stride, labels):
    """AI Hub 키포인트 → 프로젝트 CSV 변환."""
    word_mapping = build_word_mapping(dataset_root)
    label_matches = match_labels(word_mapping, labels)
    kp_dirs = find_keypoint_dirs(dataset_root)

    if not label_matches:
        print("[Convert] 매칭되는 단어가 없습니다. --scan 으로 현황을 확인하세요.")
        return

    total_saved = 0

    for label, word_nums in label_matches.items():
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        existing = len([f for f in os.listdir(save_dir) if f.endswith(".csv")])
        sample_idx = existing
        label_saved = 0

        for word_num in word_nums:
            pattern = f"WORD{word_num}_"
            matching = {d: p for d, p in kp_dirs.items() if pattern in d}

            for dirname, dirpath in matching.items():
                # 카메라 각도 필터: NIA_SL_WORD0001_REAL01_F → "F"
                angle = dirname.rsplit("_", 1)[-1]
                if angles and angle not in angles:
                    continue

                seq = load_sequence_frames(dirpath)
                if seq is None or len(seq) < SEQUENCE_LENGTH:
                    continue

                samples = make_samples(seq, SEQUENCE_LENGTH, stride)
                for sample in samples:
                    fname = os.path.join(save_dir, f"aihub_{sample_idx:04d}.csv")
                    np.savetxt(fname, sample, delimiter=",")
                    sample_idx += 1
                    label_saved += 1

        total_saved += label_saved
        if label_saved > 0:
            print(f"  [{label}] +{label_saved}개 샘플 → {save_dir}")
        else:
            print(f"  [{label}] 키포인트 데이터 없음 (WORD 매칭은 됐으나 JSON 미존재)")

    print(f"\n[Convert] 완료. 총 {total_saved}개 샘플 → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Hub 수어 키포인트 → 프로젝트 CSV 변환"
    )
    parser.add_argument(
        "--dataset", required=True, help="AI Hub 데이터셋 루트 디렉터리"
    )
    parser.add_argument(
        "--output", default="data/landmarks", help="출력 디렉터리 (default: data/landmarks)"
    )
    parser.add_argument(
        "--angles", nargs="*", default=None,
        help="카메라 각도 필터 (D F L R U). 미지정 시 전체 사용",
    )
    parser.add_argument(
        "--stride", type=int, default=15,
        help="슬라이딩 윈도우 stride (default: 15, 작을수록 샘플 많음)",
    )
    parser.add_argument(
        "--words", nargs="*", default=None,
        help="변환할 단어 목록. 미지정 시 KSL_LABELS 전체",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="변환하지 않고 매칭 현황만 출력",
    )
    args = parser.parse_args()

    target_labels = args.words if args.words else KSL_LABELS

    if args.scan:
        scan_only(args.dataset, target_labels)
    else:
        convert(args.dataset, args.output, args.angles, args.stride, target_labels)
