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

GPU 학습 서버 (`ssh -p 30007 root@cscloud.gpu3.hufs.ac.kr`, RTX 4000 Ada 20GB, CUDA 12.7, Python 3.10, 컨테이너 메모리 제한 24GB):

- [x] 종단 학습 파이프라인 — `scripts/deploy_code.sh` → `upload_dataset.sh`(MP4 제외 tar 스트림) → `server_setup.sh`(TF 2.15.1 GPU) → `run_training.sh`(변환→증강→학습→평가) → `fetch_model.sh`
- [x] 검증 실행 (2026-06-10, 예시 데이터셋 18단어×3,540샘플, `KSL_LABELS_OVERRIDE` 사용): GPU 학습 ~84 epoch, **TFLite 평가 정확도 0.98**, 추론 0.41ms, `ksl_model.tflite` 740KB (131차원 입력 (1,30,131)→(1,18) 검증 통과). 단 시연자 1명+슬라이딩 윈도우 데이터라 수치는 낙관적 — 파이프라인 동작 검증 목적
- [ ] 서버 confusion matrix의 한글 라벨이 □로 표시 (matplotlib 한글 폰트 부재 — 기능 무관)

### 1.3 미완료

- [x] KSL_LABELS 30단어 확정 (2026-06-10) — AI Hub 표제어 기준 재구성, `config/settings.py` 반영 (§1.5 WORD 목록)
- [ ] 진짜 AI Hub 데이터셋 다운로드 (§1.5의 45개 WORD 패키지) + 서버 학습
- [ ] LSTM 학습: 운용 `model/ksl_model.tflite` 미생성 (파이프라인은 검증 완료)
- [ ] 하드웨어 결선: LCD / 부저 / 스피커 / **푸시 버튼 4개**(완료 GPIO5, 페르소나 GPIO6·13·19 — README 결선표) 모두 미연결
- [ ] Gemini API 키 발급 및 `.env` 등록
- [ ] 종단 FPS 측정 (목표 ≥20 FPS)
- [ ] 1차 발표 자료 (Week 4)
- [ ] 데모 영상·최종 보고서 (Week 6)

### 1.4 입력 차원 정책 (2026-06-10 개정 — 131차원)

- KSL는 양손 언어이므로 단일 손(63차원) 입력으로는 의미 보존이 불가능.
- 입력 = **131차원 = [LEFT 21×3 | RIGHT 21×3 | 손목간 벡터 3 | presence flag 2]**. 레이아웃 단일 정의: `src/feature_format.py` (추론 `hand_tracker.py`·변환 `convert_aihub.py`가 공유 — train/serve skew 구조적 차단).
- 각 손은 손목(landmark 0) 상대좌표를 **intra-hand scale(‖landmark9−landmark0‖)로 나눠** 정규화 — 좌표계 단위(MediaPipe 0~1 vs AI Hub 미터)·카메라 거리·손 크기에 invariant. 이 scale 정규화가 없으면 AI Hub(손목상대 max≈0.09m)와 MediaPipe(max≈0.1~0.35) 분포가 어긋난다(실측).
- 손목간 벡터 = (우손목−좌손목)/두 손 scale 평균. 양손 단어의 두 손 상대 위치 신호 보존. 한 손이라도 부재면 0.
- 미감지 손은 zero + presence flag 0. 양손 모두 미감지된 프레임은 시퀀스 누락(`extract_landmarks()` None). 분류기는 연속 `NO_HAND_RESET_FRAMES`(10) 미검출 시 버퍼를 비운다(stale 추론 방지).
- MediaPipe handedness는 *입력이 거울 모드(selfie)임을 가정* — `main.py`/`collect_data.py`가 `cv2.flip(frame, 1)` 후 호출하므로 'Left' = 사용자의 해부학적 왼손. `extract_landmarks(frame, mirrored=False)`로 호출하면 라벨을 swap한다.
- 한 손 단어(예: "안녕", "주세요" 중 한 손 위주 동작)는 시연자가 자신의 **주손**으로 자연스럽게 수행한다 — 왼손잡이는 LEFT에, 오른손잡이는 RIGHT에 수집된다. `model/augment.py:flip_horizontal`이 슬롯·flag swap을 수행하므로 학습 데이터엔 양쪽 분포가 자연스럽게 채워진다. 단 flip은 **방향 의존 수어를 제외한 `FLIP_SAFE_LABELS`만 opt-in**. **억지로 비주손 시연을 강요하지 않는다**.
- handedness 신뢰도: `HANDEDNESS_SCORE_THRESHOLD`(settings.py, 기본 0.7, `.env` 오버라이드) 미만의 손은 라벨 대신 **x좌표 fallback**(거울모드에서 작은 x = 좌손)으로 슬롯 배정. 두 손이 동일 라벨로 분류되는 충돌 케이스는 score 낮은 쪽을 반대편으로 재배정하며 stderr에 warning을 출력한다.

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

**KSL_LABELS 매칭 현황 (2026-06-10 라벨 전면 개정 후):**

- 30단어 전부 AI Hub 사전 표제어와 **정확 매칭** (30/30, WORD 패키지 45개).
- 변환은 반드시 `--exact` 모드 사용 — 어근 매칭은 의미 오염을 유발한다 (실측: "기다리세요"↔"다리", "맞다"↔"뺨맞다", "먹다"↔"얻어먹다"). 이 오염 때문에 먹다·기다리세요는 라벨에서 제외하고 생활명사·감정어로 교체했다.

**현재 보유 키포인트:** WORD1501-1520 (운전면허, 골키퍼 등) — 새 KSL_LABELS와 겹침 없음 (파이프라인 검증용으로만 사용됨). 아래 45개 WORD 패키지의 키포인트를 AI Hub에서 다운로드해야 한다:

```
나(1157)  당신(1353)  좋다(0738,1191)  싫다(1278,1385)  맞다(1174,2317,2318)
가다(0943,0944,0946,1345)  오다(1149)  서다(1148,1344)  자다(1377,1378,1544)
주다(2394,2395,2396)  배고프다(0953,1197)  목마르다(2036)  아프다(1152)
피곤하다(1158)  춥다(1248)  덥다(1382)  슬프다(0009)  화나다(1236)
행복(1169)  감사(1290)  부탁(1589)  돕다(2388,2389,2390)
밥(1534)  병원(1496)  의사(0163)  엄마(1528)  가족(1492)  친구(1204)
얼마(0436)  완료(0742)
```

**변환 스크립트 사용법:**

```bash
# 매칭 현황 확인 (변환 없이 스캔만) — --exact 필수 (어근 오염 차단)
python convert_aihub.py --dataset /path/to/aihub/dataset --scan --exact

# 정면 카메라만 변환 (실 사용 환경과 유사)
python convert_aihub.py --dataset /path/to/aihub/dataset --angles F --exact

# 전체 카메라 각도 사용 (데이터 5배, 다양성 확보)
python convert_aihub.py --dataset /path/to/aihub/dataset --exact

# 슬라이딩 윈도우 stride 조정 (작을수록 샘플 많음)
python convert_aihub.py --dataset /path/to/aihub/dataset --stride 10 --exact
```

출력은 `data/landmarks/<단어>/aihub_NNNN.csv`로 저장되며, 기존 직접 촬영 데이터와 동일 디렉터리에 병합된다. `train.py`가 자동으로 인식한다.

**좌표계 차이 — 실측 확정 (2026-06-10):** AI Hub는 멀티카메라 3D 복원 좌표(미터, 비거울), MediaPipe는 0~1 정규화 좌표(거울 영상)이다. `tools/verify_aihub_alignment.py`가 동일 영상의 MP4(런타임 경로)와 키포인트를 프레임별 비교한 결과:

- 필요한 변환 = **x 부호 반전** (`convert_aihub.py:AIHUB_AXIS_SIGNS = (-1, 1, 1)`), 좌우 슬롯 swap 불필요.
- 변환 적용 후 상관계수 x=0.954, y=0.982, z=0.577 (무변환이면 x=−0.954 — 좌우 반전 데이터로 학습하게 됨).
- z 상관 0.577은 MediaPipe 단안 추정 z의 한계 — 방향 일치하는 보조 신호로 유효. ToF 카메라 불필요.
- 스케일 차이는 feature_format의 intra-hand scale 정규화가 흡수. 데이터셋 버전 변경 시 검증 스크립트를 재실행한다.

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
| 2-K | 모델이 정확해도 항상 엉뚱한 단어 출력 | sklearn `LabelEncoder`가 라벨을 유니코드 정렬 → 학습 인덱스 ≠ `KSL_LABELS[label_idx]` 역매핑 | LabelEncoder 금지, `train.py:LABEL_TO_IDX` 원본 순서 인코딩 (smoke test가 회귀 가드) |
| 2-L | TFLite 변환 시 `LLVM ERROR: Failed to infer result type(s)` abort | TF 2.16+(Keras 3)의 LSTM 변환 MLIR 버그 | `tensorflow<2.16` 고정 (2.15.1 검증) |
| 2-L′ | TFLite 변환 시 `TensorListReserve ... element_shape to be static` 또는 OOM kill(exit 137) | ① dynamic batch에서 LSTM TensorList shape 미확정 ② float16 양자화 패스의 메모리 폭주 (24GB 컨테이너 실측) | ① batch_size=1 고정 모델로 변환(`train.py:convert_and_verify_tflite`) ② float16 양자화 미적용 — 모델이 float32로도 ~740KB |
| 2-M | `AttributeError: module 'mediapipe' has no attribute 'solutions'` | mediapipe 0.10.30대부터 legacy solutions API 제거 | `mediapipe<0.10.30` 고정 |
| 2-N | AI Hub 학습 모델이 실 카메라에서 좌우 반전 동작 인식 | AI Hub는 비거울 월드좌표, 런타임은 cv2.flip 거울 영상 | `convert_aihub.py:AIHUB_AXIS_SIGNS=(-1,1,1)` x 반전 (§1.5, 실측 확정) |
| 2-O | WSL `/mnt/c` 위 venv에서 TF 업/다운그레이드 후 `No module named 'tensorflow.core'` | NTFS에서 pip 파일 교체 잔재 | `uv pip uninstall tensorflow keras` 후 재설치 |
| 2-P | `RPi.GPIO add_event_detect` → `RuntimeError: Failed to add edge detection` (RPi 실기) | 레거시 sysfs 이벤트가 Trixie의 libgpiod 커널과 충돌 | 이벤트 디텍트 사용 금지 — `src/button_input.py`는 메인 루프 폴링(에지+디바운스 자체 구현)으로 동작 |

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
- 푸시 버튼 4개: 완료=GPIO5(Pin 29), 정중=GPIO6(Pin 31), 친근=GPIO13(Pin 33), 간단=GPIO19(Pin 35). 각 버튼은 핀↔GND(Pin 30/34) 사이 연결 — 내부 풀업, 외부 저항 불필요. 완료 버튼이 문장 생성 기본 트리거 (침묵 트리거는 기본 비활성).
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
| 2026-06-10 | 이성준 + Claude (Opus 4.8) | 출력 이중화: Gemini가 (한국어, 영어) 두 줄 생성 — TTS=한국어/LCD·GUI=영어(KSL_LABELS_EN), LCD ASCII 안전망. 운용 모델(30클래스, 전수 0.95) 서버 학습·RPi 배포 완료 |
| 2026-06-10 | 이성준 + Claude (Opus 4.8) | 물리 버튼 4개(완료+페르소나 3) 도입 — src/button_input.py, 침묵 트리거 기본 비활성(SILENCE_TRIGGER_SEC=0), 페르소나 비프 1/2/3회 피드백, 결선도 재생성 |
| 2026-06-10 | 이성준 + Claude (Opus 4.8) + Codex 리뷰 | 131차원 전환(§1.4), AI Hub 축 변환 실측 확정(§1.5), LabelEncoder 버그 수정, GPU 서버 학습 파이프라인(scripts/) + 서버 E2E 검증, 함정 2-K~2-O 추가 |
| 2026-05-27 | 이성준 + Claude | §1.1 PR #7-#8 반영, §1.5 AI Hub 데이터셋 변환 파이프라인 추가, §6 convert_aihub.py 등록 |
| 2026-05-15 | 이성준 + Claude | 초안 작성 (PR #2 진행 중, Week 2 진입 시점, smoke-test 인프라 완성) |
