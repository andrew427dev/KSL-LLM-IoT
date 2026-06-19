# KSL-LLM-IoT 사용 설명서

> 실시간 한국 수어(KSL) → LLM 자연어 → 음성/LCD 출력 시스템
> 대상 사용자: 라즈베리파이 4B 위에 본 시스템을 직접 설치·운용하는 사람
> 최종 갱신: 2026-06-18

---

## 0. 한눈에 보기

```
[카메라] ─► MediaPipe Hands ─► LSTM(TFLite) ─► 단어 버퍼
                                                    │
                                                    ▼
                                        [완료] 또는 3초 침묵
                                                    │
                                                    ▼
                                            Gemini 2.5 Flash
                                                    │
                                          ┌─────────┴─────────┐
                                          ▼                   ▼
                                    [I2C LCD 20x4]        [TTS 스피커]
```

- **인식 단위**: 30프레임(약 1초) 시퀀스 → 한 단어
- **문장화 트리거**: **완료 버튼**(GPIO5) 또는 단어 "완료" 수행 (3초 무동작 트리거는 기본 비활성)
- **문체 선택**: 버튼 3개 — 정중·친근·간단 (비프 1/2/3회로 확인)
- **신호음**: 단어가 인식될 때마다 부저 짧게 1회

---

## 1. 준비물

### 1.1 하드웨어

| 부품 | 사양 | 비고 |
|------|------|------|
| Raspberry Pi 4B | RAM 4GB 이상 | OS: Raspberry Pi OS (Bookworm/Trixie, 64-bit aarch64) |
| 카메라 | Pi Camera v2 **또는** USB 웹캠 | 자동 감지(`src/main.py:CameraReader`) |
| I2C LCD | 20×4, 주소 `0x27` 기본 | `LCD_I2C_ADDRESS` 환경변수로 변경 가능 |
| 부저 | Active 부저 | GPIO17(BCM) 기본 — `BUZZER_PIN` |
| 푸시 버튼 ×4 | 모멘터리 택트 스위치 | 문장 완료 GPIO5(물리 29) / 정중 GPIO6(31) / 친근 GPIO13(33) / 간단 GPIO19(35). 각 버튼은 핀과 GND(물리 6) 사이 연결 — 내부 풀업, 외부 저항 불필요 |
| 스피커 | USB-오디오형: USB만 · 증폭(3.5mm)형: 3.5mm(오디오)+USB/외부(전원) | TTS 출력용 |
| microSD | 32GB 이상 (Class 10) | OS + 모델 + 데이터 |
| 전원 | 5V/3A USB-C | 카메라+스피커 동시 사용 시 필수 |

> 결선도: [`docs/wiring_diagram.png`](docs/wiring_diagram.png)
> (`docs/generate_wiring_diagram.py`로 재생성 가능)

### 1.2 소프트웨어

- Python **3.11**
- `requirements.txt` (개발/학습용 PC)
- `requirements-rpi.txt` (라즈베리파이 실시간 추론용)
- **Gemini API Key** (Google AI Studio에서 발급)
- I2C 활성화: `sudo raspi-config` → Interface Options → I2C → Enable

---

## 2. 최초 설치 (라즈베리파이)

```bash
# 1) 저장소 클론
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git
cd KSL-LLM-IoT

# 2) 가상환경 (uv + Python 3.11)
#    RPi OS Trixie 기본 Python은 3.13이며 mediapipe aarch64 휠이 없어 venv 생성이 실패한다.
#    반드시 uv로 Python 3.11 가상환경을 만든다 (시스템 Python은 그대로 둔다).
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11
source .venv/bin/activate

# 3) 의존성 (RPi에서는 RPi 전용 파일 사용)
uv pip install -r requirements-rpi.txt

# 4) 환경변수 파일 생성
cp .env.example .env
nano .env       # GEMINI_API_KEY 등 채우기

# 5) 학습된 모델 배치
#    PC에서 학습한 model/ksl_model.tflite 를 그대로 같은 경로에 복사
```

### 2.1 `.env` 필수 항목

| 키 | 의미 | 기본값 |
|----|------|--------|
| `GEMINI_API_KEY` | Gemini 2.5 Flash API 키 (필수) | — |
| `MODEL_PATH` | TFLite 모델 경로 | `model/ksl_model.tflite` |
| `CONFIDENCE_THRESHOLD` | 단어 확정 임계 신뢰도 (0~1) | `0.85` |
| `SEQUENCE_LENGTH` | 시퀀스 프레임 수 | `30` |
| `SILENCE_TRIGGER_SEC` | 무동작 자동 문장화(초). 3초 무동작 시 자동 발화(완료 버튼과 병행). `0` = 비활성(버튼 전용) | `3.0` |
| `DUPLICATE_FILTER_SEC` | 같은 단어 재인식 차단 창(초) | `3.0` |
| `LCD_I2C_ADDRESS` | LCD 주소 | `0x27` |
| `BUZZER_PIN` | 부저 GPIO(BCM) | `17` |
| `BUTTON_COMPLETE_PIN` | 문장 완료 버튼 GPIO(BCM) | `5` |
| `BUTTON_PERSONA_POLITE_PIN` | 정중 문체 버튼 GPIO(BCM) | `6` |
| `BUTTON_PERSONA_FRIENDLY_PIN` | 친근 문체 버튼 GPIO(BCM) | `13` |
| `BUTTON_PERSONA_BRIEF_PIN` | 간단 문체 버튼 GPIO(BCM) | `19` |
| `BUTTON_DEBOUNCE_MS` | 버튼 디바운스(ms) | `300` |
| `CAMERA_INDEX` | OpenCV 카메라 인덱스 | `0` |
| `HANDEDNESS_SCORE_THRESHOLD` | 좌/우 손 판별 신뢰도 임계값 (미달 시 위치 기반 배정) | `0.7` |
| `NO_HAND_RESET_FRAMES` | 연속 미검출 시 인식 버퍼 초기화 프레임 수 | `10` |
| `KSL_LABELS_OVERRIDE` | (학습 실험 전용) 단어 목록 대체 — 운용 시 비움 | (비움) |
| `SENTENCE_PERSONA` | 문장 생성 문체 — `정중`/`친근`/`간단` | `정중` |

> LCD 주소가 불명확한 경우 `i2cdetect -y 1`로 I2C 버스 상의 장치를 확인한다.

---

## 3. 일상 사용 — "수어로 말하기"

### 3.1 시스템 시작

```bash
cd ~/KSL-LLM-IoT
source .venv/bin/activate
python src/main.py
```

부팅 후 LCD에 다음이 표시되면 준비 완료입니다.
```
KSL-LLM-IoT Ready
Show your sign...
```

### 3.2 동작 순서

1. 카메라 정면에 손이 보이도록 위치합니다.
2. 등록된 30개 수어 단어 중 하나를 **약 1초간** 수행합니다.
3. 인식 시 **부저음** + LCD에 단어와 신뢰도가 표시됩니다.
4. 단어를 이어서 수행하면 버퍼에 누적됩니다. 같은 단어는 3초(`DUPLICATE_FILTER_SEC`) 이내 재인식이 무시된다 — 동일 단어를 연속 입력하려면 3초 이상 간격을 둔다.
5. **완료 버튼**(GPIO5)을 누르면 Gemini가 자연어 문장을 만들어 LCD/스피커로 출력한다. "완료" 수어도 동일하게 동작한다. **3초 이상 무동작 시에도 자동으로 문장이 생성된다**(기본 `SILENCE_TRIGGER_SEC=3.0`, 완료 버튼과 병행 — 버튼 전용을 원하면 `.env`에서 `0`으로 설정).
6. **출력 분리**: 음성(TTS)은 한국어 문장, LCD·화면 표시는 영어다 — 문자형 LCD(HD44780)는 한글을 렌더링하지 못한다. 인식 단어도 LCD에는 영어 라벨(예: 배고프다→hungry)로 표시된다.
7. **문체 버튼** 3개로 출력 문체를 즉시 전환한다 — 정중(비프 1회) / 친근(비프 2회) / 간단(비프 3회). 비프 횟수로 화면을 보지 않아도 적용 문체를 확인할 수 있다. 시작 문체는 `.env` `SENTENCE_PERSONA`. 모니터 연결(GUI) 모드에서는 `SPACE` 키 = 완료, `p` 키 = 문체 순환.

### 3.3 등록 단어 (총 30개)

| 분류 | 단어 |
|------|------|
| 인칭·응답 | 나, 당신, 좋다, 싫다, 맞다 |
| 동사 | 가다, 오다, 서다, 자다, 주다 |
| 상태 | 배고프다, 목마르다, 아프다, 피곤하다, 춥다, 덥다, 슬프다, 화나다 |
| 감정·의사소통 | 행복, 감사, 부탁, 돕다 |
| 생활 명사 | 밥, 병원, 의사, 엄마, 가족, 친구, 얼마 |
| 시스템 | **완료**(전송 트리거) |

단어 목록은 AI Hub 수어 영상 데이터셋의 사전 표제어와 일치한다 (2026-06-10 개정).

> 각 단어의 표준 동작은 [국립국어원 한국수어사전](https://sldict.korean.go.kr) 에서 검색해 참고한다.

### 3.4 종료

- 디버그 창(`KSL-LLM-IoT`)이 떠 있는 상태에서 **`q`** 키
- 또는 터미널에서 `Ctrl+C`

---

## 4. 데이터 수집 (개발자 모드)

> 새 단어를 추가하거나 인식 정확도를 올리려면 본인 데이터를 수집해 재학습합니다.

```bash
# config/settings.py 의 KSL_LABELS 에 단어가 등록되어 있어야 함
python collect_data.py --word 안녕 --samples 100
```

조작:
- **SPACE** : 수집 시작/정지 토글
- **q** : 종료

- 시퀀스 길이는 `SEQUENCE_LENGTH`(기본 30프레임) 단위로 자동 저장됩니다.
- 저장 위치: `data/landmarks/<단어>/0000.csv`, `0001.csv`, ...
- 이어찍기 지원: 기존 파일 개수부터 번호가 이어집니다.

### 4.1 데이터 수집 기준

- 단어당 100 샘플 이상 수집 (`SEQUENCE_LENGTH=30` 기준 30프레임 × 100 = 3,000프레임).
- 조명·각도·배경 조건을 변경하며 수집 (모델 일반화에 필요).
- 카메라와의 거리: 40~80 cm (Pi Camera v1 640×480 기준 손이 프레임 내 적정 크기로 잡히는 범위).

---

## 5. 모델 학습 & 배포

```bash
# 1) (선택) 증강 — 원본 1개당 3개 생성
python model/augment.py --factor 3

# 2) LSTM 학습
python model/train.py

# 3) (선택) 평가 — 정확도·혼동 행렬·추론 지연
python model/evaluate.py

# 4) 산출물: model/ksl_model.tflite
#    → 라즈베리파이의 같은 경로에 복사 (전송 방법은 §5.1 참고)
```

학습 → 평가 → 배포 흐름은 **PC(또는 GPU 서버)에서 학습**, **RPi에서 추론**이 원칙입니다.

학습 로그의 `Test Accuracy`는 학습에 사용되지 않은 시연자(held-out)의 원본 데이터 기준 수치다 — 새 사용자에 대한 일반화 성능을 나타낸다. `model/evaluate.py`는 학습이 기록한 `model/holdout.json`의 동일 집합만 평가하며, 이 파일이 없으면 학습 데이터가 포함된 전수 평가로 동작하므로 수치가 실제보다 높게 나온다 (로그에 경고 출력).

---

### 5.1 PC ↔ 라즈베리파이 파일 전송 수단

PuTTY 설치 시 같은 디렉터리(`C:\Program Files\PuTTY\`)에 `pscp.exe`(scp 클라이언트)와 `psftp.exe`(sftp 클라이언트)가 함께 설치된다. 별도 설치 없이 호출 가능하다. GUI 환경에서는 별도로 WinSCP를 설치해 사용할 수 있다.

본 프로젝트에서 전송 대상이 되는 파일은 다음과 같다.

| 방향 | 파일 |
|------|------|
| PC → RPi | `src/*.py`, `config/settings.py`, `model/ksl_model.tflite`, `.env` |
| RPi → PC | `data/landmarks/<단어>/*.csv` (학습용 회수) |

#### ① 라즈베리파이 IP 확인

RPi에서:
```bash
hostname -I            # 예: 192.168.0.50
```

#### ② pscp — 단발 명령으로 전송

Windows PowerShell 또는 cmd에서 실행한다 (PuTTY 창이 아니다).

```cmd
:: 단일 파일 (PC → RPi)
pscp "C:\Users\DOCTOR\Desktop\coding\college_4-1\iot_system\KSL-LLM-IoT\src\main.py" tocomboy@192.168.0.50:/home/tocomboy/Desktop/KSL-LLM-IoT/src/main.py

:: 디렉터리 재귀 (-r)
pscp -r "C:\...\KSL-LLM-IoT\src" tocomboy@192.168.0.50:/home/tocomboy/Desktop/KSL-LLM-IoT/

:: 학습 모델 (PC → RPi)
pscp "C:\...\KSL-LLM-IoT\model\ksl_model.tflite" tocomboy@192.168.0.50:/home/tocomboy/Desktop/KSL-LLM-IoT/model/

:: 역방향 (RPi → PC)
pscp -r tocomboy@192.168.0.50:/home/tocomboy/Desktop/KSL-LLM-IoT/data/landmarks "C:\...\KSL-LLM-IoT\data\"
```

PATH에 pscp가 없을 경우 풀경로 호출:
```cmd
"C:\Program Files\PuTTY\pscp.exe" <원본> <대상>
```

#### ③ psftp — 대화형 세션으로 여러 파일 전송

```cmd
psftp tocomboy@192.168.0.50
```

```
psftp> cd /home/tocomboy/Desktop/KSL-LLM-IoT/src
psftp> lcd C:\...\KSL-LLM-IoT\src
psftp> put main.py
psftp> put hand_tracker.py classifier.py
psftp> get ../data/landmarks/안녕/0000.csv
psftp> quit
```

#### ④ WinSCP — GUI 클라이언트

[WinSCP](https://winscp.net) 설치 후 첫 실행 시 PuTTY 저장 세션을 가져온다. 두 패널(PC / RPi) 사이 파일을 드래그 앤 드롭으로 전송한다.

#### ⑤ 공개키 인증 (비밀번호 입력 생략)

PC에서 SSH 키 쌍을 PuTTY 포맷(`.ppk`) 또는 OpenSSH 포맷으로 생성하고 공개키를 RPi `~/.ssh/authorized_keys`에 등록한다. 등록 후 `pscp`/`psftp`/PuTTY가 비밀번호 입력 없이 접속한다. 자동화 스크립트에서 사용한다.

#### ⑥ git을 통한 전송

PC에서 `git push` → RPi에서 `git pull`로 소스 파일을 동기화한다. 단 `.gitignore`에 포함된 `data/`, `model/*.tflite`, `.env`는 git으로 전송되지 않으므로 ②~④ 방식과 병행한다.

#### ⑦ PuTTY 한글 인코딩 설정

PuTTY → Window → Translation → "Remote character set"을 `UTF-8`로 설정하면 콘솔에 한글이 정상 표시된다 (`?덈뀞` → `안녕`).

---

## 6. 문제 해결 (Troubleshooting)

| 증상 | 원인 후보 | 대응 |
|------|-----------|------|
| `No camera available (all backends failed)` | OpenCV/rpicam-vid/picamera2 셋 다 실패 | `rpicam-hello --timeout 2000`로 OS 단 카메라 확인 → 결선·`config.txt`의 `camera_auto_detect=1` 점검 |
| `rpicam-vid backend skipped` | 5초 안에 첫 YUV 프레임 못 받음 | 다른 프로세스가 카메라 점유 중인지 확인(`pgrep -a rpicam`), 다른 카메라 앱 종료 후 재실행 |
| 콘솔에 한글이 `?덈뀞` 처럼 깨짐 | PuTTY 인코딩 미설정 | PuTTY → Window → Translation → Remote character set = **UTF-8** |
| LCD에 아무것도 안 뜸 | I2C 비활성화 / 주소 다름 | `sudo raspi-config`로 I2C ON, `i2cdetect -y 1` |
| `RPi.GPIO not available` 메시지 | 비-RPi 환경에서 실행 중 | 정상 — PC 개발 시 부저·버튼 비활성, GUI 키(`SPACE`/`p`)로 대체 |
| 버튼을 눌러도 반응 없음 | 배선이 GND 아닌 3.3V/5V에 연결됨 / 핀 번호 불일치 | 버튼은 GPIO핀↔GND 연결(눌림=LOW). `.env`의 `BUTTON_*_PIN`(BCM)과 실제 배선 일치 확인 |
| 버튼 1회 눌렀는데 여러 번 동작 | 스위치 채터링 | `BUTTON_DEBOUNCE_MS` 증가 (기본 300) |
| 스피커에서 재생과 무관하게 "툭툭"·웅웅 잡음 | 증폭 스피커가 RPi USB 5V의 부하 리플을 증폭 (전원 리플, 실측 확인) | 스피커 USB 전원을 RPi가 아닌 **별도 충전기/보조배터리**에 연결. 배경 "쉬-" 잡음은 `config.txt`의 `audio_pwm_mode=2`, `disable_audio_dither=1`로 완화 |
| 단어 인식이 자꾸 틀림 | 학습 데이터 부족/편향 | 본인 환경에서 추가 데이터 수집 → 재학습 |
| 시작 직후 `모델 shape 불일치` 오류로 종료 | 구버전 모델 파일 또는 단어 목록(`KSL_LABELS`)이 다른 모델 | 최신 `model/ksl_model.tflite`를 재배포(§5.1) 하거나 현재 단어 목록으로 재학습 |
| Gemini 응답 없음 | API 키 미설정 / 쿼터 초과 | `.env`의 `GEMINI_API_KEY` 확인, 네트워크 점검 |
| TTS 무음 | 스피커 라우팅 | `raspi-config` → System Options → Audio |
| FPS가 낮음 | 해상도 과대 | `.env`의 `CAMERA_WIDTH/HEIGHT` 낮추기 |
| 같은 단어가 연속 들어감 | 정상 — 3초 중복 필터 | `.env`의 `DUPLICATE_FILTER_SEC` 조정 |

---

## 7. 평가 지표 (목표값)

| 항목 | 목표 |
|------|------|
| 단어 인식 정확도 | ≥ 85% |
| RPi 4B 추론 FPS | ≥ 20 FPS |
| Gemini 응답 지연 | ≤ 2 초 |
| 종단(end-to-end) 지연 | ≤ 4 초 |

---

## 8. 안전·주의사항

- **카메라 영상은 로컬에서만 처리**됩니다. Gemini로 전송되는 것은 *인식된 단어 텍스트*뿐입니다.
- API 키(`GEMINI_API_KEY`)는 절대 깃에 커밋하지 마세요 — `.env`는 `.gitignore` 적용 대상입니다.
- 부저/LCD 결선 작업 시 RPi 전원을 차단한 상태에서 진행한다.

---

## 9. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-06-19 | 결선도(§1.1) 갱신 — 증폭 스피커는 **USB(전원)+3.5mm(오디오) 둘 다 연결**해야 동작(USB만/3.5mm만으로는 무음, 실측 확인). 부저 라벨 겹침·버튼 공통 GND 선 표기 보정 |
| 2026-06-18 | 무동작 자동 문장화 기본값 `SILENCE_TRIGGER_SEC=3.0`으로 변경(§2.1·§3.2) — 완료 버튼과 병행(침묵 자동 발화). 버튼 전용을 원하면 `.env`에서 `0`. RPi 설치 절차를 `uv venv --python 3.11` 기준으로 통일(§2) |
| 2026-06-12 | 결선도 갱신(§1.1) — 카메라를 USB 웹캠 기준으로 표기, 버튼 GND를 물리 6으로 정정(실측) |
| 2026-06-12 | 중복 인식 필터 1.5→3.0초, `.env` `DUPLICATE_FILTER_SEC`로 조정 가능(§2.1·§3.2) — 동작 유지 시 같은 단어 반복 누적 방지 |
| 2026-06-11 | 학습 `Test Accuracy`가 held-out 시연자 기준으로 변경(§5) — 일반화 성능 지표. 구버전·단어 불일치 모델은 시작 시 `모델 shape 불일치` 오류로 즉시 종료(§6) |
| 2026-06-10 | 출력 분리 — 음성은 한국어, LCD·화면은 영어(LCD 한글 미지원). 인식 단어 영어 라벨 표 적용 |
| 2026-06-10 | 물리 버튼 4개 도입 (§1.1 결선) — 문장 완료 버튼이 기본 트리거가 되고 무동작 자동 문장화(`SILENCE_TRIGGER_SEC`)는 기본 비활성. 문체 버튼 3개(비프 1/2/3회 피드백) 추가 |
| 2026-06-10 | 문장 생성 문체(페르소나) 기능 추가 — `.env` `SENTENCE_PERSONA`=정중/친근/간단, GUI 모드 `p` 키 순환 (§2.1, §3.2) |
| 2026-06-10 | 등록 단어 30개 전면 개정 (§3.3) — AI Hub 데이터셋 표제어 기준. 기존 단어 중 17개 유지, 13개 교체. **이전 단어로 학습된 모델·수집 데이터와 호환되지 않음** |
| 2026-06-10 | 입력 포맷을 131차원(양손 + 손 크기 정규화 + 손목간 벡터 + presence flag)으로 전환 — **기존 .tflite 모델·수집 CSV는 재학습/재수집 필요**. AI Hub 데이터셋 변환(`convert_aihub.py`)이 동일 포맷으로 출력. `.env` 키 3개 추가(아래 §2.1) |
| 2026-05-14 | Gemini SDK를 `google-genai`로 마이그레이션 (`google-generativeai`는 EOL). 메인 인식 루프 블로킹 제거 — `SentenceBuilder`의 Gemini 호출, LCD I2C 쓰기, 부저 펄스를 모두 워커 스레드로 분리. 인식 직후 카메라 정지 시간이 사라짐 |
| 2026-05-14 | `rpicam-vid` 서브프로세스 카메라 백엔드 추가 — Pi Camera v1/v2를 venv에서 잡지 못하던 문제 해소. SSH/PuTTY용 헤드리스 모드(`KSL_HEADLESS=1`) 도입. PuTTY 파일 전송 가이드(§5.1) 추가 |
| 2026-05-14 | 초기 작성 — 30단어, Gemini 2.5 Flash, 카메라 자동감지 반영 |

---

## 10. 도움말 / 참고

- 개발자용 빌드·구조 설명 : [`README.md`](README.md)
- 기여 가이드 : [`CONTRIBUTING.md`](CONTRIBUTING.md)
- 결선도 : [`docs/wiring_diagram.png`](docs/wiring_diagram.png)
- 팀: 이성준(202102467) · 배진규(202001647) — HUFS IoT Systems 2026 봄
