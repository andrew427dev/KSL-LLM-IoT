# HANDOFF — 2026-05-27

본 문서는 KSL-LLM-IoT 프로젝트 작업을 이어받는 팀원·차기 세션이 *현재 상태와 다음 단계*를 파악하기 위해 작성된다. 매번의 작업 마무리 시 갱신한다.

- 작성 시점: **2026-05-27** (Week 4 진입)
- 작성자: 이성준 + Claude (Opus 4.7)
- 대상: 배진규(팀원), 이후 본인, 차기 세션

---

## 1. 현재 상태 (2026-05-27 기준)

### 1.1 코드

- 기본 브랜치: `main` (보호) ← `develop` ← feature 브랜치 (GitFlow 변형)
- **현재 작업 브랜치: `develop`** (HEAD: `d3ce0c0`)
- 머지 완료 PR:

  | PR | 브랜치 | 내용 |
  |----|--------|------|
  | #2 | `feat/smoke-test-pipeline` | smoke-test 인프라, 비동기 파이프라인, genai SDK |
  | #5 | `docs/data-sources` | 데이터 소스 섹션 확장 |
  | #6 | `develop` → `main` | develop 통합 머지 |
  | #7 | `feat/two-hand-input` | 양손 입력 전환 (INPUT_SHAPE 63→126) |
  | #8 | `ci/static-checks` | 정적 검증 워크플로 추가 |

- 머지 흐름: feature 브랜치 → `develop` → `main`.

### 1.2 검증된 동작

Raspberry Pi 4B (Raspberry Pi OS Trixie, aarch64) + Pi Camera v1 (OV5647) 환경:

- [x] `rpicam-vid` 서브프로세스 카메라 백엔드
- [x] MediaPipe 손 인식
- [x] 분류기 더미 모드 (`model/ksl_model.tflite` 부재 시)
- [x] SentenceBuilder 오프라인 모드 (`GEMINI_API_KEY` 부재 시)
- [x] 헤드리스 실행 (`KSL_HEADLESS=1`)
- [x] 비동기 파이프라인 — 인식 직후 카메라 루프 정지 없음
- [x] `Ctrl+C` 안전 종료

### 1.3 미완료

- [ ] 데이터 수집: 직접 촬영 + AI Hub 변환 병행 (§1.5 참조)
- [ ] LSTM 학습: `model/ksl_model.tflite` 미생성
- [ ] 하드웨어 결선: LCD / 부저 / 스피커 모두 미연결
- [ ] Gemini API 키 발급 및 `.env` 등록
- [ ] 종단 FPS 측정 (목표 ≥20 FPS)
- [ ] 1차 발표 자료 (Week 4)
- [ ] 데모 영상·최종 보고서 (Week 6)

### 1.4 입력 차원 정책 (2026-05-16 확정)

- KSL는 양손 언어이므로 단일 손(63차원) 입력으로는 의미 보존이 불가능.
- 입력 = **126차원 = [LEFT 21×3 | RIGHT 21×3]**. 각 손은 자신의 손목 기준 정규화.
- 미감지 손은 zero-pad. 양손 모두 미감지된 프레임은 시퀀스 누락(`extract_landmarks()` None).
- MediaPipe handedness는 *입력이 거울 모드(selfie)임을 가정* — `main.py`/`collect_data.py`가 `cv2.flip(frame, 1)` 후 호출하므로 'Left' = 사용자의 해부학적 왼손.
- 한 손 단어(예: "안녕", "주세요" 중 한 손 위주 동작)는 시연자가 자신의 **주손**으로 자연스럽게 수행한다 — 왼손잡이는 LEFT에, 오른손잡이는 RIGHT에 수집된다. `model/augment.py:flip_horizontal`이 LEFT↔RIGHT 블록 swap을 수행하므로 학습 데이터엔 양쪽 분포가 자연스럽게 채워진다. **억지로 비주손 시연을 강요하지 않는다** (부자연스러운 동작이 학습 분포에 들어가는 것을 막기 위함).
- handedness 신뢰도: `src/hand_tracker.py:HANDEDNESS_SCORE_THRESHOLD = 0.7` 미만의 손은 미감지로 처리. 두 손이 동일 라벨로 분류되는 충돌 케이스는 score 낮은 쪽을 반대편으로 재배정하며 stderr에 warning을 출력한다. Week 2 수집 중 raw score 분포를 보고 임계값을 0.6~0.8 범위에서 재조정 가능.

### 1.5 AI Hub 수어 데이터셋 변환 파이프라인 (2026-05-27)

직접 촬영 외에 [AI Hub 수어 영상 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103)의 3D 키포인트를 프로젝트 CSV로 변환하여 학습 데이터를 보강할 수 있다.

**데이터셋 구조:**

| 경로 | 내용 | 필요 여부 |
|------|------|-----------|
| `라벨링데이터/REAL/WORD/*_keypoint/` | 프레임별 3D 키포인트 JSON | **필요** |
| `수어 영상/.../morpheme/` | WORD번호 → 한국어 단어 매핑 | **필요** |
| `원천데이터/REAL/WORD/` (MP4) | 원본 영상 | 불필요 (키포인트 이미 추출됨) |

**AI Hub 3D 키포인트 → 프로젝트 호환성:**

| 항목 | AI Hub | 프로젝트 |
|------|--------|----------|
| 손 관절 수 | 21개 | 21개 (동일) |
| 3D 좌표 | (x, y, z, confidence) | (x, y, z) |
| 정규화 | 절대 좌표 (멀티카메라 3D 복원) | 손목 기준 상대 좌표 |

변환 시 confidence를 버리고 (x,y,z)만 추출한 뒤 손목 기준 정규화를 적용한다. `hand_tracker.py:_normalize_hand()`와 동일한 로직이다.

**KSL_LABELS 매칭 현황 (3,000개 단어 중):**

- 정확 매칭: 14개 — 나, 당신, 좋다, 싫다, 맞다, 가다, 오다, 서다, 자다, 배고프다, 목마르다, 아프다, 피곤하다, 완료
- 어근 매칭: 5개 — 감사합니다(←감사), 먹다(←먹), 행복하다(←행복), 얼마예요(←얼마), 기다리세요(←기다리다)
- 매칭 없음: 11개 — 안녕, 미안합니다, 반갑습니다, 우리, 네, 아니오, 마시다, 앉다, 도와주세요, 주세요, 화장실

**현재 보유 키포인트:** WORD1501-1520 (운전면허, 골키퍼 등) — KSL_LABELS와 겹침 없음. 아래 WORD번호에 해당하는 키포인트 패키지를 AI Hub에서 추가 다운로드해야 한다:

```
WORD0738(좋다)  WORD0742(완료)  WORD0943(가다)  WORD0953(배고프다)
WORD1148(서다)  WORD1149(오다)  WORD1152(아프다) WORD1157(나)
WORD1158(피곤하다) WORD1169(행복)  WORD1174(맞다)  WORD1278(싫다)
WORD1290(감사)  WORD1353(당신)  WORD1377(자다)  WORD2036(목마르다)
```

**변환 스크립트 사용법:**

```bash
# 매칭 현황 확인 (변환 없이 스캔만)
python convert_aihub.py --dataset /path/to/aihub/dataset --scan

# 정면 카메라만 변환 (실 사용 환경과 유사)
python convert_aihub.py --dataset /path/to/aihub/dataset --angles F

# 전체 카메라 각도 사용 (데이터 5배, 다양성 확보)
python convert_aihub.py --dataset /path/to/aihub/dataset

# 슬라이딩 윈도우 stride 조정 (작을수록 샘플 많음)
python convert_aihub.py --dataset /path/to/aihub/dataset --stride 10
```

출력은 `data/landmarks/<단어>/aihub_NNNN.csv`로 저장되며, 기존 직접 촬영 데이터와 동일 디렉터리에 병합된다. `train.py`가 자동으로 인식한다.

**좌표계 차이 주의:** AI Hub는 멀티카메라 3D 복원 좌표(미터 단위), MediaPipe는 0~1 정규화 좌표이다. 손목 기준 상대 좌표 변환으로 스케일 차이를 흡수하지만, 학습 시 직접 촬영 데이터와 혼합 비율을 조절하며 검증 정확도를 모니터링해야 한다.

---

## 2. 환경 제약 — 함정과 해결법

다음 항목들은 *학기 진행 중 실제로 부딪혀서 통과한* 사항이다. 작업 재개 시 같은 함정을 재현하지 않도록 한다.

| # | 증상 / 함정 | 원인 | 해결 |
|---|------------|------|------|
| 2-A | `pip install mediapipe`가 `No matching distribution found` | Python 3.13 + aarch64 조합에 휠 없음 (2026-05 시점) | `uv venv --python 3.11`로 Python 3.11 강제 |
| 2-B | RPi OS Trixie 기본 Python이 3.13 | 시스템 업그레이드 흐름 | `uv`가 prebuilt 3.11 다운로드 — 시스템 Python은 그대로 둠 |
| 2-C | `apt install python3.11` 실패 | Debian Trixie 기본 저장소에서 제외됨 | apt 사용 안 함, `uv` 사용 |
| 2-D | `python-prctl` 빌드 실패 (`libcap`) | picamera2 의존성 컴파일 시 헤더 필요 | `sudo apt install libcap-dev` |
| 2-E | `ModuleNotFoundError: libcamera` | `python3-libcamera` apt 패키지는 시스템 Python 3.13 ABI 전용 | picamera2 사용하지 않고 `rpicam-vid` 서브프로세스 백엔드로 우회 (이미 구현됨) |
| 2-F | `libcamerify python ...`로 OpenCV가 `read()` 시 빈 프레임 | LD_PRELOAD V4L2 셈과 OpenCV 호환성 한계 | 사용 금지. `rpicam-vid` 백엔드만 사용 |
| 2-G | `ValueError: No API key` at `genai.Client()` | google-genai 신 SDK가 생성 시점에 검증 | `SentenceBuilder.__init__`에서 `GEMINI_API_KEY` 빈 값 체크 → `_offline=True` (이미 구현됨, `345f11e`) |
| 2-H | 콘솔에 한글 깨짐 (`?덈뀞`) | PuTTY 기본 인코딩이 UTF-8 아님 | PuTTY → Window → Translation → Remote character set = **UTF-8** |
| 2-I | 인식 직후 카메라 정지 (1~3초) | Gemini API + LCD I2C + beep이 메인 루프에서 동기 실행 | 워커 스레드로 분리 (`cdb2ac2`) |
| 2-J | `gh pr edit` title 갱신 안 됨 | GitHub의 Projects(classic) deprecation 경로 충돌 | `gh api -X PATCH /repos/.../pulls/N`으로 직접 호출 |

---

## 3. 작업 재개 절차

### 3.1 PC (개발 + 학습)

```bash
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git
cd KSL-LLM-IoT
# PR #2 머지 전: feat/smoke-test-pipeline / 머지 후: develop
git checkout feat/smoke-test-pipeline

# uv + Python 3.11
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
cp .env.example .env
```

상세는 README §A 참조.

### 3.2 RPi (배포 + 실행)

```bash
# 시스템 패키지
sudo apt update
sudo apt install -y i2c-tools rpicam-apps libcap-dev libcamera-dev \
                    pkg-config python3-dev build-essential

# 저장소 가져오기 (또는 PC에서 pscp 전송)
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git ~/Desktop/KSL-LLM-IoT
cd ~/Desktop/KSL-LLM-IoT
git checkout feat/smoke-test-pipeline

# venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-rpi.txt

# 환경설정
cp .env.example .env
nano .env   # GEMINI_API_KEY는 공란 가능 (offline fallback)

# 카메라/I2C 검증
rpicam-hello --timeout 2000
sudo raspi-config       # → Interface Options → I2C → Enable → 재부팅
i2cdetect -y 1

# 실행
KSL_HEADLESS=1 python src/main.py
```

상세는 README §B 참조.

### 3.3 동작 상태 매트릭스

| `GEMINI_API_KEY` | `model/ksl_model.tflite` | 결과 |
|---|---|---|
| 설정됨 | 존재 | 정상 운영 |
| 설정됨 | 없음 | LLM 동작, 분류기 더미 모드 |
| 비어있음 | 존재 | 분류기 동작, 문장은 단어 나열 |
| 비어있음 | 없음 | 전 구간 fallback (현재 기본 상태) |

각 모드 모두 파이프라인 전체가 끊김 없이 실행된다. 부분 실패가 전체 정지로 이어지지 않는 구조다.

---

## 4. 다음 작업

### 4.1 Week 2 (5/14~5/21) — 데이터 수집

위치: PC + 웹캠.

```bash
python collect_data.py --word 안녕 --samples 100
# SPACE = 녹화 토글, q = 종료
# 출력: data/landmarks/<단어>/0000.csv ...
```

- 30단어 전체에 대해 반복. `config/settings.py:KSL_LABELS` 참조.
- 단어당 100샘플 미만이면 학습 일반화 부족 위험. 가능하면 단어당 150~200샘플.
- 조명·각도·배경을 변경하며 수집.
- 카메라 거리 40~80cm.
- 분담안 예시: 이성준 15단어, 배진규 15단어. 분담은 팀 회의에서 확정.

### 4.2 Week 3 (5/21~5/28) — 학습

위치: PC.

```bash
python model/augment.py --factor 3      # 선택
python model/train.py                   # → model/ksl_model.tflite
python model/evaluate.py                # 정확도 확인
```

목표: 단어 인식 정확도 ≥85%. 미달 시 추가 데이터 수집 후 재학습.

생성된 `.tflite`를 RPi로 전송 (`pscp` 또는 git LFS, USER_MANUAL §5.1).

### 4.3 하드웨어 결선 (병렬 가능)

위치: RPi 본체. 코드 변경 불필요.

README §🔌 Hardware Connection 참조. 핵심:

- I2C LCD: SDA=GPIO2(Pin 3), SCL=GPIO3(Pin 5), VCC=5V(Pin 2), GND(Pin 6). 주소 `0x27` 기본.
- 부저: 신호=GPIO17(Pin 11), GND(Pin 9).
- 스피커: 3.5mm 또는 USB.
- I2C 활성화: `sudo raspi-config` → Interface Options → I2C → Enable → 재부팅.

검증: `i2cdetect -y 1`로 LCD 주소 표시되면 OK.

### 4.4 Week 4 (5/28~6/1) — 통합 + 1차 발표

- 실제 모델 + 하드웨어로 종단 테스트.
- FPS 측정 (목표 ≥20 on RPi 4B).
- 1차 발표 자료 작성.

### 4.5 Week 5 (6/1~6/10) — Gemini API

- https://aistudio.google.com 에서 API 키 발급 (무료 티어로 본 프로젝트 사용량 충당).
- RPi `.env`에 `GEMINI_API_KEY=AIza...` 입력 (인용부호 없이).
- 재실행 시 오프라인 모드 안내 메시지가 사라지고 LLM 경로로 전환.
- 시스템 프롬프트 튜닝(`config/settings.py:GEMINI_SYSTEM_PROMPT`)으로 출력 톤 조정.
- 응답 지연 ≤4초 측정.

### 4.6 Week 6 (6/10~6/22) — 데모 + 보고서

- 데모 영상 (인사·동사·요청 시나리오 각 1개 이상).
- 최종 보고서 (구조·결과·한계·향후 개선).
- 발표 자료.
- main 브랜치 안정화, README·USER_MANUAL §9 changelog 최신화.

---

## 5. 책임 분담 (현재 기록)

| 영역 | 담당 | 비고 |
|------|------|------|
| 카메라·MediaPipe·분류기 인프라 | 이성준 | 완료 (PR #2) |
| 비동기 파이프라인·SDK 마이그레이션 | 이성준 | 완료 (PR #2) |
| 문서화 (README/USER_MANUAL/CLAUDE/HANDOFF) | 이성준 | 진행 중 |
| 데이터 수집 30단어 | 미정 | Week 2 회의에서 분담 확정 |
| 모델 학습 | 미정 | Week 3 |
| 하드웨어 결선 | 미정 | 병렬 진행 |
| Gemini 프롬프트 튜닝 | 미정 | Week 5 |
| 데모 영상 촬영 | 미정 | Week 6 |

배진규 담당 영역이 결정되면 본 표를 갱신한다.

---

## 6. 핵심 파일·문서

| 파일 | 내용 |
|------|------|
| [README.md](README.md) | 프로젝트 소개, 환경 설치(PC/RPi 분리), 하드웨어 결선표, 운영 모드 매트릭스 |
| [USER_MANUAL.md](USER_MANUAL.md) | 운용자용 한국어 매뉴얼. §5.1 PuTTY 파일 전송, §6 트러블슈팅, §9 changelog |
| [CLAUDE.md](CLAUDE.md) | AI 에이전트 가이드. §1.1 매뉴얼 갱신 트리거, §1.4 모든 문서 공통 작성 규칙 (객관 어조) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 기여 가이드 |
| [docs/wiring_diagram.png](docs/wiring_diagram.png) | 결선도 (재생성: `python docs/generate_wiring_diagram.py`) |
| `src/main.py` | 메인 루프, `CameraReader` 3단 폴백, 헤드리스 감지 |
| `src/sentence_builder.py` | google-genai 호출 + 오프라인 fallback + 비동기 큐 |
| `src/classifier.py` | TFLite 추론 + 더미 모드 |
| `src/lcd_display.py` | I2C LCD + 워커 스레드 큐 |
| `make_dummy_model.py` | 검증용 더미 TFLite 생성 (PC) |
| `collect_data.py` | 랜드마크 CSV 수집 — 직접 촬영 (PC) |
| `convert_aihub.py` | AI Hub 수어 키포인트 JSON → 프로젝트 CSV 변환 (§1.5 참조) |

---

## 7. 다음 핸드오프까지의 체크리스트

작업 마무리 시점에 다음을 갱신:

- [ ] §1.1 PR 번호·커밋·머지 상태
- [ ] §1.2 검증된 동작 (새 항목 체크박스)
- [ ] §1.3 미완료 (완료된 항목 제거 또는 체크박스 마크)
- [ ] §2 함정과 해결법 — 새로 부딪힌 함정 1행 추가
- [ ] §4 다음 작업 — 진척된 Week 섹션 갱신
- [ ] §5 분담 — 확정/변경 사항 반영
- [ ] §8 변경 이력 본 문서 한 줄 추가

---

## 8. 본 문서 변경 이력

| 날짜 | 작성자 | 내용 |
|------|--------|------|
| 2026-05-27 | 이성준 + Claude | §1.1 PR #7-#8 반영, §1.5 AI Hub 데이터셋 변환 파이프라인 추가, §6 convert_aihub.py 등록 |
| 2026-05-15 | 이성준 + Claude | 초안 작성 (PR #2 진행 중, Week 2 진입 시점, smoke-test 인프라 완성) |
