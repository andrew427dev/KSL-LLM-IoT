# Final Report Outline — KSL-LLM-IoT
# (PDF ≥10 pages, due 2026-06-22, file: IoT_LeeSungjoon_202102467_TermProject.rar)

본 문서는 보고서 작성용 소스 아웃라인이다. 각 절에 들어갈 내용 포인트,
그림/표 목록, 본 저장소에서 실측·확정된 수치를 정리한다.
[TBD] 표시는 실데이터 학습/실기 데모 후 채운다.

---

## 제출 요건 체크리스트

- [ ] PDF report ≥ 10 pages (영문 권장 — PPT·발표가 영어 필수)
- [ ] PPT ≥ 15 slides, in English (`docs/presentation_outline.md` 참조)
- [ ] Presentation 8 min + 2 min Q&A (max 10 min, in English)
- [ ] Demo video (short, 동작 증명 — "very important")
- [ ] 1 zipped folder: Demo + PDF + PPT → `IoT_LeeSungjoon_202102467_TermProject.rar`

필수 포함 섹션: Project Idea / System Design & Methodology /
Hardware & Software Details / Results & Contributions / Future Work

---

## 1. Project Idea (1–1.5 pages)

**Main concept (영문 초안):**
> A real-time Korean Sign Language (KSL) translator running entirely on a
> Raspberry Pi 4B edge device. A camera captures two-hand signing; MediaPipe
> extracts 3D hand landmarks; a TFLite LSTM classifies 30 KSL words; a large
> language model (Gemini 2.5 Flash) composes the recognized words into natural
> Korean sentences, delivered through a speaker (TTS) and an I2C LCD.
> Physical buttons provide an accessible, latency-free control interface.

포인트:
- 문제: 수어 사용자의 일상 의사소통 장벽 — 통역 인력 의존
- 해결: 엣지(저비용 RPi) + 클라우드 LLM 하이브리드 — 인식은 로컬(프라이버시·지연), 문장 생성만 API
- IoT 특성: 카메라/GPIO 버튼(입력) → 엣지 추론 → 부저·LED·LCD·스피커(출력), 클라우드 학습-엣지 배포 사이클
- 차별점: ① 단어 나열이 아닌 LLM 자연어 문장 ② 사용자 선택형 문체(persona)
  ③ 버튼+부저/LED(청각·시각) 동기 피드백의 접근성 설계 ④ 공개 데이터셋(AI Hub)과 런타임 카메라 간
  좌표계 정렬을 실측으로 검증한 train/serve 일관성

## 2. System Design & Methodology (2.5–3 pages)

**그림 목록:**
- Fig.1 전체 파이프라인 (README의 다이어그램 확장):
  Camera → MediaPipe(2 hands) → 131-dim feature → LSTM(TFLite) → word buffer
  → [complete button | "완료" sign] → Gemini → TTS + LCD
- Fig.2 131-dim feature layout (feature_format.py 도식)
- Fig.3 학습 데이터 흐름: AI Hub keypoints → axis transform → shared
  normalization → CSV → augment → train(GPU server) → TFLite → RPi

**핵심 방법론 (수치 포함):**
1. **Two-hand 131-dim representation** — [L 21×3 | R 21×3 | wrist-to-wrist 3 |
   presence 2]. 손목 상대좌표를 intra-hand scale(‖lm9−lm0‖)로 정규화 →
   카메라 거리·손 크기·좌표계 단위에 invariant.
2. **Train/serve consistency by construction** — 추론(hand_tracker)과 데이터
   변환(convert_aihub)이 단일 모듈(feature_format.py)을 공유.
3. **Empirical axis alignment** — 동일 영상의 MP4(런타임 경로)와 AI Hub
   3D keypoint를 프레임별 비교(쌍 725개): 최적 변환 x-flip,
   상관계수 x=0.954 / y=0.982 / z=0.577. 무변환 시 x=−0.954
   (좌우 반전 학습 — 본 검증으로 차단). z=0.577은 단안 추정 z의 한계지만
   방향 일치 보조 신호 → 깊이 카메라(ToF) 불필요 근거.
4. **LSTM classifier** — 2×LSTM(128,64)+Dense, ~196k params, 30 classes,
   input (30,131). float32 TFLite 740KB.
5. **LLM sentence generation** — word list → Gemini 2.5 Flash, persona
   (polite/friendly)별 system prompt, 1회 호출로 두 페르소나 동시 생성·캐시
   (토큰 절약, 전환 시 0 호출), async worker(카메라 루프 비차단), offline fallback.
6. **Methodology of robustness** — confidence 0.85 + 3.0s dedup,
   10-frame no-hand buffer reset, handedness x-fallback, flip 증강의
   방향 의존 수어 opt-out.

## 3. Hardware & Software Details (2.5–3 pages)

**표: H/W 구성** (README 결선표 재사용)
| Component | Interface | Pin |
|---|---|---|
| USB webcam (/dev/video0, demo) — Pi Camera v1/CSI also supported | USB / CSI | — |
| I2C LCD 20×4 (0x27) | I2C | GPIO2/3 |
| Active buzzer | GPIO out | GPIO17 |
| Status LED (mirrors buzzer, via resistor) | GPIO out | GPIO22 |
| Push buttons ×4 (complete + undo + persona×2) | GPIO in (internal pull-up) | GPIO5/6/13/19 ↔ GND |
| Speaker | 3.5mm/USB | — |

- Fig.4 결선도 `docs/wiring_diagram.png`
- 버튼 회로: 내부 풀업 + GND 스위칭 (전류 ~66µA, 외부 저항 불필요, 안전 근거)
- 부저+LED 동기 점멸 1/2회 = 페르소나(정중/친근) 확인 (LED = 농인용 시각 피드백, 접근성). 완료 = 길게 1회, 페르소나 전환·완료 재누름 = 캐시 문장 재생. undo 버튼 = 단어 입력 중 마지막 단어 제거 / 문장 출력 후 현재 문체 재생성

**표: S/W 스택**
| Layer | Tech | 비고 |
|---|---|---|
| CV | MediaPipe Hands 0.10.x (<0.10.30), OpenCV | legacy solutions API |
| Edge inference | tflite-runtime | RPi, Python 3.11 (uv) |
| Training | TensorFlow 2.15.1 GPU | 클라우드 서버 RTX 4000 Ada |
| LLM | google-genai (Gemini 2.5 Flash) | persona system prompt |
| OS/IoT | Raspberry Pi OS Trixie, RPi.GPIO(polling), smbus2 | |

**구현 디테일 (코드 설명 요구사항 대응):**
- `src/feature_format.py` 핵심 함수 코드 발췌 + 설명
- `src/button_input.py` 폴링 에지 검출 (Trixie에서 event-detect 불가 → 폴링 전환 사례)
- `src/sentence_builder.py` async worker + persona config 재구성
- 학습 파이프라인 스크립트 5종 (deploy/download/setup/train/fetch) — 클라우드-엣지 워크플로
- CI (GitHub Actions): 의존성 없는 회귀 테스트 30종

## 4. Results & Contributions (2–2.5 pages)

**정량 결과:**
| 항목 | 값 |
|---|---|
| Pipeline validation (예시 18 classes, 3,540 samples) | TFLite accuracy 0.98 |
| 운용 모델 (30 classes, 시연자 16명, AI Hub) | TFLite held-out accuracy **0.72** (0.7168; held-out 시연자 REAL01·02, 2,595 시퀀스 — 학습 76,460) |
| RPi 실기 인식 (USB 웹캠, `diagnose_live`) | 27/30 단어 정상, 확신 분포 0.3~0.9 (잔존 혼동: 밥/배고프다/주다) |
| TFLite inference latency | 0.41 ms (server CPU) / [TBD] ms (RPi) |
| End-to-end FPS on RPi 4B | 6.4 (MediaPipe 추론 병목 — 카메라 단독 30; target ≥20 미달, 시간 리샘플링으로 정합) |
| Model size | 740 KB |
| Sentence latency (Gemini) | [TBD] s (target ≤4) |

- Fig.5 confusion matrix
- Fig.6 [TBD] 데모 사진 (LCD 출력 + 버튼)

**Contributions (보고서 강조 포인트):**
1. 좌표계 정렬의 실측 검증 방법론 — 공개 키포인트 데이터셋과 런타임 추출기의
   체계적 비교 (재사용 가능한 도구 `tools/verify_aihub_alignment.py`)
2. Train/serve 단일 소스 전처리 설계로 skew 구조적 차단
3. 치명 결함 발견·수정 사례: LabelEncoder 유니코드 정렬(라벨 역매핑 불일치),
   TF 2.16 LSTM-TFLite 변환 불가, float16 양자화 OOM — 재현 조건과 해법 문서화
4. 접근성 중심 인터페이스: 인식율 의존 트리거 제거(물리 버튼), 청각(부저)+시각(LED 동기) 피드백 — 농인 사용자 시각 신호, 완료 길게·재생
5. 완전 자동화된 cloud-train → edge-deploy 파이프라인

## 5. Future Work (0.5–1 page)

- AI Hub 문장(SEN) 데이터셋으로 연속 수어(continuous signing) 인식 확장
- 어휘 확장 (30 → 수백 단어): 라벨-표제어 정합 파이프라인은 이미 일반화됨
- 광FOV USB 카메라 교체 검토 (FOV 54° 제약 — 코드 변경 0으로 교체 가능 설계)
- 상체 pose keypoint 추가 — 위치 의존 수어("나" vs "당신") 분리도 향상
- 온디바이스 경량 LLM (Gemma 등) — 완전 오프라인 동작
- 한글 출력 LCD(graphic LCD) 또는 모바일 컴패니언 앱

## 6. 부록 후보

- 30단어 목록 (USER_MANUAL §3.3)
- 트러블슈팅 사례집 (HANDOFF §2 — 함정 2-A~2-Z, 27건)
- 저장소·PR 이력 (#2~#9)

---

## 데모 영상 샷 리스트 (short demo video — "important")

1. (5s) 시스템 부팅 — LCD "KSL-LLM-IoT Ready"
2. (15s) 수어 3단어 연속 인식 — 단어마다 비프 + LCD 표시 ("나", "배고프다", "밥")
3. (5s) **완료 버튼** 누름 → LCD/스피커로 자연어 문장 출력
4. (15s) **페르소나 버튼** 전환(비프 2회=친근) 후 같은 단어열 → 다른 문체 문장
5. (10s) 손 미검출 → 버퍼 리셋 동작(오인식 방지) 시연
6. (10s) 하드웨어 클로즈업 — 버튼 4개, LCD, 카메라
총 ~60s. 화면 녹화(LCD)와 외부 촬영(전체 모습) 교차 편집.
