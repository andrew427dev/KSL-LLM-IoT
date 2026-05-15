# CLAUDE.md — KSL-LLM-IoT 프로젝트 규칙

> 이 파일은 Claude Code(및 OMC 에이전트)가 이 저장소에서 작업할 때 따라야 할
> **프로젝트-한정 규칙**만 담는다. 글로벌 규칙(한국어 응답, 증거 기반, 중복
> 파일 읽기 금지 등)은 사용자 ~/.claude/CLAUDE.md 에 이미 정의되어 있다.

---

## 0. 프로젝트 개요 (1줄 요약)

라즈베리파이 4B에서 카메라 → MediaPipe → LSTM(TFLite) → Gemini 2.5 Flash → TTS/LCD 로 이어지는 실시간 한국 수어(KSL) 번역기.

---

## 1. 사용자 매뉴얼(USER_MANUAL.md) 갱신 규칙 (필수)

본 저장소에는 사용자가 직접 참조하는 `USER_MANUAL.md`가 존재한다. 아래 §1.1 표의 파일/항목이 변경되면 동일 커밋(또는 직후 커밋)에서 `USER_MANUAL.md`도 갱신한다.

### 1.1 갱신 트리거 (이 파일들이 바뀌면 매뉴얼도 갱신)

| 변경된 파일/항목 | 매뉴얼에서 반영할 위치 |
|------------------|------------------------|
| `config/settings.py` 의 `KSL_LABELS` | §3.3 등록 단어 표 |
| `config/settings.py` 의 `TRIGGER_WORD`, `SILENCE_TRIGGER_SEC`, `DUPLICATE_FILTER_SEC`, `CONFIDENCE_THRESHOLD`, `SEQUENCE_LENGTH` | §0 한눈에 보기, §2.1 `.env` 표, §3.2 동작 순서 |
| `config/settings.py` 의 `BUZZER_PIN`, `LCD_I2C_ADDRESS`, `LCD_NUM_COLS/ROWS`, `CAMERA_*` | §1.1 하드웨어 표, §2.1 `.env` 표 |
| `config/settings.py` 의 `GEMINI_MODEL` | §0 한눈에 보기, §9 변경 이력 |
| `.env.example` 항목 추가/삭제 | §2.1 `.env` 필수 항목 표 |
| `requirements.txt` / `requirements-rpi.txt` 의 주요 패키지 | §1.2 소프트웨어 |
| `src/main.py` 의 키바인딩(`q` 등), CLI 인자, 카메라 자동감지 로직, `CameraReader` 백엔드(opencv/rpicam/picamera2), `KSL_HEADLESS` 환경변수, `beep()` 동작 | §3.1·§3.4·§1.1 카메라 행·§6 트러블슈팅 |
| `src/sentence_builder.py` 의 공개 API(`add_word`/`check_silence_trigger`/`poll_sentence`/`get_buffer_preview`), Gemini SDK 패키지명/`Client` 호출 방식 | §0 한눈에 보기 (파이프라인), §2.1 `.env` 표, §9 changelog (SDK 변경은 운용자에게 가시) |
| `src/lcd_display.py` 의 공개 메서드 시그니처, 비동기 큐 정책 | §6 트러블슈팅 (LCD 무반응 케이스) |
| 파일 전송 워크플로(scp/pscp/psftp/WinSCP/git pull) 변경 | §5.1 PuTTY로 파일 전송 |
| `collect_data.py` 의 키바인딩(SPACE/q), 인자명 | §4 데이터 수집 |
| `model/train.py`, `model/augment.py`, `model/evaluate.py` 의 CLI 인자 | §5 모델 학습 & 배포 |
| `docs/wiring_diagram.png` 재생성 | §1.1 비고 행, §10 참고 |
| 새로운 트러블슈팅 케이스 발견 (이슈/디버그 세션) | §6 표에 한 줄 추가 |

### 1.2 갱신 절차

1. 코드 변경 후 **항상** `USER_MANUAL.md`의 어느 §가 영향받는지 점검한다.
2. 영향 § 본문을 갱신하고, **§9 변경 이력 표 맨 위에 한 줄**을 추가한다.
   - 날짜는 절대 표기(예: `2026-05-14`).
   - 내용은 사용자가 *행동을 바꿔야 하는* 변경 중심으로 간결히 (한 줄).
3. README.md 와 USER_MANUAL.md 가 **상충하지 않는지** 확인한다.
   - README = 개발자/소개용 / USER_MANUAL = 운용자용.
   - 동일 사실(예: 단어 수, 핀 번호)은 두 곳이 일치해야 한다.
4. 매뉴얼의 코드 블록은 **실제로 실행 가능한 명령**만 남긴다 — 추측 금지.

### 1.3 매뉴얼에 *추가하지 말 것*

- 내부 구현 디테일(클래스 다이어그램, 코드 스니펫 등) → `README.md` 또는 `docs/`로.
- 미구현/계획 단계 기능 → 매뉴얼은 **현재 동작하는 것만**.
- 비밀값/실제 API 키 — 절대 안 됨.

### 1.4 모든 문서(README / USER_MANUAL / CONTRIBUTING / docs/) 공통 작성 규칙

본 저장소는 2인 팀 프로젝트(이성준·배진규)이며 학기 평가 산출물이다. 모든 문서는 **객관적·기술적 어조**로 작성한다. 다음 표현은 사용하지 않는다.

| 금지 유형 | 예시 (사용 금지) | 대체 |
|-----------|-----------------|------|
| 주관적 비교/순위 | "가장 빠른", "가장 깔끔한", "최선의", "추천" | 각 방식의 *특성·트레이드오프*를 사실로 나열 |
| 근거 없는 권장 | (스펙·측정 근거 없는) "권장", "권유" | 수치/근거 명시 또는 해당 표현 제거 |
| 개인 선호 가정 | "~하기 싫으면", "~를 선호하면", "편한 선택" | 조건문으로 환경 기술 ("GUI 환경에서는") |
| 비격식 어미 | "유용해요", "편해요", "~예요", "~이에요" | 평서형 "~이다", "~한다" |
| 친밀어/감정 | "여러분", "당신(데이터 제외)", "다행히", "안타깝게도" | 무주어 또는 "사용자/운용자" |
| 권유형 어미 | "스캔하세요", "확인해보세요", "~하시면" | 명령형 "스캔한다", "확인한다" |

선택지가 여럿이면 *각 선택지의 동작·요구사항·제약*만 나열하고, 어느 것이 더 낫다는 평가는 적지 않는다. 평가자/팀원이 자체 기준으로 판단할 수 있도록 한다.

표·코드 블록의 라벨(예: KSL 단어 목록의 "당신", "주세요")은 *데이터*이므로 위 규칙의 예외다.

---

## 2. 코드 변경 규칙

### 2.1 하드웨어/환경 분기
- `src/main.py:27-34`처럼 `RPi.GPIO` 임포트는 **try/except로 감싸 PC 개발이 가능**해야 한다. 이 규약을 깨지 말 것.
- 카메라는 OpenCV → picamera2 순서로 자동 폴백한다(`CameraReader`). 새 카메라 백엔드 추가 시 같은 패턴 유지.

### 2.2 설정 우선순위
- 하드코딩 금지. 모든 튜닝 값은 `config/settings.py` → `.env` 오버라이드 가능하게.
- `.env.example`에 해당 키를 같이 추가하고 USER_MANUAL.md §2.1 표도 갱신.

### 2.3 KSL_LABELS 수정 시
- 새 라벨 추가 = 데이터 재수집 + 모델 재학습 필요. **단순 라벨만 추가하고 끝내지 말 것** — 변경 이력에 학습 필요 여부를 명시.
- `TRIGGER_WORD`("완료")는 항상 `KSL_LABELS`의 마지막 원소로 유지.

### 2.4 테스트
- 글로벌 규칙에 따라 코드 변경 시 관련 테스트 갱신. 테스트 위치는 `tests/`.
- 라즈베리파이 전용 코드(`RPi.GPIO`, `picamera2`)는 mock 또는 skip 처리.

---

## 3. 디렉터리 & 산출물

| 경로 | 용도 | 깃 커밋 여부 |
|------|------|--------------|
| `data/raw/` | 원본 영상 | ✗ |
| `data/landmarks/<단어>/*.csv` | 랜드마크 시퀀스 | ✗ (크기 큼) |
| `data/augmented/` | 증강 데이터 | ✗ |
| `model/ksl_model.tflite` | 배포 모델 | △ (릴리스 태그 시 LFS) |
| `.env` | 비밀값 | ✗ (절대 금지) |
| `USER_MANUAL.md` | 사용자 매뉴얼 | ✓ |
| `README.md` | 개발자 소개 | ✓ |

---

## 4. 커밋 메시지

- 글로벌 규칙(한국어 본문 허용, 제목은 영어 또는 한국어)에 따른다.
- USER_MANUAL.md 와 코드가 함께 변경된 커밋의 제목은 가능하면 `docs:` 접두어 대신 변경 본질을 따른다 — 예: `feat: add 화장실 sign + manual update` 처럼 매뉴얼 동기화가 동일 커밋에서 일어났음을 본문에 명시.

---

## 5. 자주 쓰는 명령

```bash
# 개발 PC (학습)
python collect_data.py --word <단어> --samples 100
python model/augment.py --factor 3
python model/train.py

# 라즈베리파이 (운용)
python src/main.py
i2cdetect -y 1       # LCD 주소 확인
ls /dev/video*       # 카메라 디바이스 확인
```

---

## 6. 참고

- USER_MANUAL.md §9 변경 이력은 **사용자에게 보이는 changelog**다. 내부 리팩터링은 적지 말고, 사용자 행동에 영향이 있는 것만.
- 의심스러우면: "이 변경으로 사용자가 켜는 방법·조작·기대 결과가 달라지나?" → YES면 매뉴얼 갱신, NO면 README 또는 코드 주석.
