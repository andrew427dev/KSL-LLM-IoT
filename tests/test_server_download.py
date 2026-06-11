"""
test_server_download.py
scripts/server_download_aihub.sh 무결성 검증 — PR #9 리뷰 #2.

배경: unzip 오류를 `2>/dev/null || true`로 마스킹한 채 zip 무조건 삭제 +
done 마커 무조건 생성 → 손상 아카이브·디스크 풀에서 클래스가 무음 누락되고
재실행도 스킵되는 결함이 있었다. 수정 후 계약:
  - unzip rc 0(성공)·11(무매칭 정상)·1(경고) 허용, 그 외 치명 → exit 1
  - 치명 시 zip 보존 + done 마커 미생성 (재실행 가능)
  - 한 zip에서 45개 WORD 전부 무매칭(0건)이면 구조 이상 — 치명 처리

실행: aihubshell 스텁(픽스처 zip을 WORK로 복사)을 PATH에 주입해
스크립트를 실제로 구동한다. 외부 의존 없음 — CI 실행 가능.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "server_download_aihub.sh")

AIHUBSHELL_STUB = """#!/usr/bin/env bash
# aihubshell 스텁 — 픽스처 zip을 현재 디렉터리(WORK)로 복사한다
cp "$AIHUBSHELL_FIXTURE" .
"""

# 시스템 unzip 부재 환경(개발 WSL 등)용 셔임 — 실제 unzip의 종료 코드 계약
# (0=성공, 3=손상 아카이브, 11=패턴 무매칭)을 모사한다. 서버에는 실제 unzip이
# 있으므로 셔임은 시스템 unzip이 없을 때만 테스트 PATH에 설치된다.
UNZIP_SHIM = '''#!/usr/bin/env python3
import fnmatch
import os
import sys
import zipfile

forced = os.environ.get("UNZIP_SHIM_FORCE_RC")
if forced:
    sys.exit(int(forced))

args = sys.argv[1:]
dest = "."
if "-d" in args:
    i = args.index("-d")
    dest = args[i + 1]
    del args[i:i + 2]
paths = [a for a in args if not a.startswith("-")]
zpath = paths[0]
pattern = paths[1] if len(paths) > 1 else None

try:
    zf = zipfile.ZipFile(zpath)
    names = zf.namelist()
except (zipfile.BadZipFile, OSError):
    sys.stderr.write("unzip-shim: cannot read archive\\n")
    sys.exit(3)

members = [n for n in names if pattern is None or fnmatch.fnmatch(n, pattern)]
if not members:
    sys.exit(11)
zf.extractall(dest, members)
sys.exit(0)
'''


def _make_zip(path, inner_word="1157"):
    """WORD{inner_word} 키포인트 디렉터리 1개를 담은 정상 zip."""
    name = f"NIA_SL_WORD{inner_word}_REAL01_F"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}/{name}_000001_keypoints.json", "{}")


def _run_script(tmp, fixture_zip, force_rc=None):
    """스텁 환경에서 스크립트를 시연자 1명 모드로 실행한다.

    force_rc: 지정 시 unzip 셔임을 강제 설치하고 모든 unzip 호출이 해당
    rc로 종료하게 한다 — 디스크 풀(50) 등 재현 불가능한 치명 분기 검증용.
    """
    dest = os.path.join(tmp, "dest")
    work = os.path.join(tmp, "work")
    os.makedirs(dest, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    # 형태소 단계는 이번 검증 범위 밖 — 완료 마커로 스킵
    open(os.path.join(dest, ".morpheme_done"), "w").close()

    keyfile = os.path.join(tmp, "aihub_key")
    with open(keyfile, "w") as f:
        f.write("dummy-key")

    stub_dir = os.path.join(tmp, "bin")
    os.makedirs(stub_dir, exist_ok=True)
    stub = os.path.join(stub_dir, "aihubshell")
    with open(stub, "w") as f:
        f.write(AIHUBSHELL_STUB)
    os.chmod(stub, 0o755)

    if shutil.which("unzip") is None or force_rc is not None:
        shim = os.path.join(stub_dir, "unzip")
        with open(shim, "w") as f:
            f.write(UNZIP_SHIM)
        os.chmod(shim, 0o755)

    env = os.environ.copy()
    env.update({
        "DEST": dest,
        "WORK": work,
        "KEY_FILE": keyfile,
        "AIHUBSHELL_FIXTURE": fixture_zip,
        "PATH": stub_dir + os.pathsep + env["PATH"],
    })
    if force_rc is not None:
        env["UNZIP_SHIM_FORCE_RC"] = str(force_rc)
    proc = subprocess.run(
        ["bash", SCRIPT, "1"],
        env=env, capture_output=True, text=True, timeout=60,
    )
    return proc, dest, work


def _work_zips(work):
    return [f for f in os.listdir(work) if f.endswith(".zip")]


def test_healthy_zip_extracts_and_marks_done():
    """정상 zip → 추출 성공·zip 삭제·done 마커. 나머지 44개 WORD의
    무매칭(rc 11)은 치명으로 오판하지 않아야 한다."""
    with tempfile.TemporaryDirectory() as tmp:
        fixture = os.path.join(tmp, "real_word_keypoint.zip")
        _make_zip(fixture, inner_word="1157")
        proc, dest, work = _run_script(tmp, fixture)

        assert proc.returncode == 0, f"정상 zip인데 실패: {proc.stdout}\n{proc.stderr}"
        assert os.path.exists(os.path.join(dest, ".signer_1_done")), "done 마커 부재"
        assert not _work_zips(work), "처리 완료된 zip이 삭제되지 않음"
        extracted = os.path.join(
            dest, "라벨링데이터", "REAL", "WORD", "NIA_SL_WORD1157_REAL01_F"
        )
        assert os.path.isdir(extracted), "키포인트 디렉터리 미추출"
    print("[PASS] test_healthy_zip_extracts_and_marks_done")


def test_corrupt_zip_fails_and_preserves_state():
    """손상 zip → exit≠0·zip 보존·done 마커 미생성 (재실행 가능 상태 유지)."""
    with tempfile.TemporaryDirectory() as tmp:
        fixture = os.path.join(tmp, "real_word_keypoint.zip")
        with open(fixture, "wb") as f:
            f.write(b"this is not a zip archive")
        proc, dest, work = _run_script(tmp, fixture)

        assert proc.returncode != 0, "손상 zip인데 성공으로 처리됨"
        assert not os.path.exists(os.path.join(dest, ".signer_1_done")), \
            "손상인데 done 마커 생성 — 재실행이 스킵된다"
        assert _work_zips(work), "손상 zip이 삭제됨 — 원인 조사 불가"
    print("[PASS] test_corrupt_zip_fails_and_preserves_state")


def test_zero_match_zip_fails():
    """45개 WORD 전부 무매칭인 zip → 구조 이상으로 치명 처리."""
    with tempfile.TemporaryDirectory() as tmp:
        fixture = os.path.join(tmp, "real_word_keypoint.zip")
        _make_zip(fixture, inner_word="9999")  # KSL 45개 목록에 없는 WORD
        proc, dest, work = _run_script(tmp, fixture)

        assert proc.returncode != 0, "0개 매칭 zip인데 성공으로 처리됨"
        assert not os.path.exists(os.path.join(dest, ".signer_1_done"))
        assert _work_zips(work), "의심 zip이 삭제됨"
    print("[PASS] test_zero_match_zip_fails")


def test_fatal_unzip_rc_fails_and_preserves_state():
    """디스크 풀(rc=50) 등 0/1/11 외 rc → 치명: exit≠0·zip 보존·마커 미생성.
    실환경에서 재현 불가능한 rc라 셔임 강제 주입으로 분기를 검증한다."""
    with tempfile.TemporaryDirectory() as tmp:
        fixture = os.path.join(tmp, "real_word_keypoint.zip")
        _make_zip(fixture, inner_word="1157")
        proc, dest, work = _run_script(tmp, fixture, force_rc=50)

        assert proc.returncode != 0, "rc=50인데 성공으로 처리됨"
        assert not os.path.exists(os.path.join(dest, ".signer_1_done"))
        assert _work_zips(work), "치명 오류인데 zip이 삭제됨"
    print("[PASS] test_fatal_unzip_rc_fails_and_preserves_state")


if __name__ == "__main__":
    test_healthy_zip_extracts_and_marks_done()
    test_corrupt_zip_fails_and_preserves_state()
    test_zero_match_zip_fails()
    test_fatal_unzip_rc_fails_and_preserves_state()
    print("\nAll tests done.")
