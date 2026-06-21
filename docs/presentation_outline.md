# Presentation Outline — KSL-LLM-IoT
# (PPT in English, ≥15 slides, talk 8 min + 2 min Q&A, due 2026-06-22)

슬라이드별 제목·핵심 내용·비주얼. 발표 시간 배분 합계 ≈ 8분.
[TBD]는 실데이터 학습·실기 데모 후 수치 확정.

---

## Slide 1 — Title (10s)
**KSL-LLM-IoT: Real-Time Korean Sign Language Translation on an Edge Device**
- Team, student IDs, course, date
- Visual: 시스템 전체 사진 (RPi + 카메라 + LCD + 버튼)

## Slide 2 — Problem & Motivation (30s)
- Deaf/hard-of-hearing people face daily communication barriers
- Human interpreters: expensive, not always available
- Goal: an affordable, always-on, privacy-preserving translator
- Visual: 문제 상황 일러스트 1장

## Slide 3 — Project Idea (30s)
- Sign words → camera → on-device recognition → LLM → natural sentence → voice + display
- Edge AI (recognition stays local) + Cloud LLM (only word lists sent)
- One-line pitch: "Sign it, press the button, hear the sentence."
- Visual: 5단계 아이콘 플로우

## Slide 4 — System Architecture (40s)
- Full pipeline diagram: Camera → MediaPipe (2 hands) → 131-dim feature
  → LSTM (TFLite) → word buffer → [button / sign trigger] → Gemini 2.5 Flash
  → TTS + I2C LCD
- Async workers: camera loop never blocks on LLM/LCD/buzzer
- Visual: Fig.1 (보고서와 공용)

## Slide 5 — Hardware (40s)
- Raspberry Pi 4B, Pi Camera v1, I2C LCD 20×4, buzzer, status LED, speaker
- **4 push buttons** (complete + undo + 2 persona) — internal pull-up,
  button-to-GND wiring, no external resistors
- Visual: wiring_diagram.png + 실물 사진

## Slide 6 — Why Buttons? Accessibility by Design (30s)
- Recognition-latency triggers (silence timer) = unpredictable errors
- Physical buttons: deterministic, tactile, usable without watching the screen
- Buzzer + synchronized LED blink 1/2 = which persona is active (LED = visual cue for deaf/HoH); complete = long beep+LED, replay cached sentence on persona switch / re-press; undo button = remove last word (while entering) or re-generate the sentence (after output)

## Slide 7 — Feature Engineering: 131-dim Two-Hand Input (40s)
- [LEFT 63 | RIGHT 63 | wrist-to-wrist 3 | presence 2]
- Wrist-relative ÷ intra-hand scale → invariant to camera distance,
  hand size, and coordinate units
- Single shared module for training & inference → no train/serve skew
- Visual: Fig.2 레이아웃 도식

## Slide 8 — Training Data: AI Hub Sign-Language Dataset (30s)
- 3,000-word public dataset, multi-view 3D keypoints, professional signers
- We match our 30-word vocabulary to dataset headwords exactly
  (root-matching caused semantic contamination — e.g., "기다리세요" ↔ "다리")
- Server-side downloader fetches only the 45 needed WORD packages

## Slide 9 — Key Finding: Coordinate Alignment (40s) ★기술 하이라이트
- Dataset = world coordinates (meters, non-mirrored);
  runtime = MediaPipe image coordinates (mirrored)
- Measured frame-by-frame on the same videos (725 pairs):
  best transform = **x-flip**; correlation x=0.954, y=0.982, z=0.577
- Without it: x = **−0.954** → model would learn left-right–inverted signs
- Visual: 상관계수 막대그래프 + 비교 도식

## Slide 10 — Model & Cloud Training Pipeline (40s)
- LSTM(128→64) + Dense, ~196k params, TFLite 740 KB (fits the edge)
- One-command pipeline on GPU server (RTX 4000 Ada):
  download → convert → augment → train → evaluate → fetch
- Augmentation: mirror-flip (direction-safe words only), Gaussian noise
  with presence masking, non-linear time warp

## Slide 11 — Bugs We Caught (and How) (40s) ★엔지니어링 신뢰성
- LabelEncoder Unicode sorting → training/inference index mismatch
  (model "correct" but outputs wrong words) — caught by cross-review
- TF 2.16 LSTM→TFLite MLIR crash; float16 quantization OOM → TF 2.15 + float32
- RPi.GPIO edge-detect broken on latest kernel → polling-based buttons
- All guarded by 30+ dependency-free CI regression tests

## Slide 12 — LLM Sentence Generation with Personas (30s)
- Word list → Gemini 2.5 Flash with persona-specific system prompt
- 2 styles selectable by button: polite / friendly (casual speech)
- One call generates both styles and caches them (token-saving) — persona switch = 0 extra calls
- Demo example: "나 배고프다 밥" →
  polite: "저는 배가 고파서 밥을 부탁드립니다."
  friendly: "나 배고픈데 밥 줘."
- Offline fallback when no API key/network

## Slide 13 — Results (50s)
- Pipeline validation: 18 classes / 3,540 samples → TFLite accuracy 0.98
- Production model (30 classes, 16 signers): **72%** held-out accuracy
  (TFLite, 2 signers held out), confusion matrix
- On-device recognition: 27/30 words, confidence 0.3–0.9 (healthy spread)
- Inference 0.41 ms (server) / [TBD] ms (RPi); model 740 KB
- End-to-end FPS on RPi: 6.4 (MediaPipe-bound; time-resampling aligns the window)
- Visual: confusion matrix + 지표 표

## Slide 14 — Live/Video Demo (60s) ★"works correctly" 증명
- Demo video (~60s): sign 3 words → complete button → spoken sentence
  → persona switch → different style
- Backup: 사전 녹화 영상 필수 (현장 데모 실패 대비)

## Slide 15 — Contributions (30s)
1. Empirical dataset↔runtime alignment methodology (reusable tool)
2. Train/serve consistency by single-source preprocessing
3. Accessibility-first interface (buttons + auditory buzzer + synchronized visual LED for deaf/HoH)
4. Fully automated cloud-train → edge-deploy workflow

## Slide 16 — Future Work (20s)
- Continuous signing (sentence dataset), larger vocabulary
- Wide-FOV USB camera; upper-body pose features
- On-device LLM for full offline operation

## Slide 17 — Q&A (predicted questions)
- Why not ToF/depth camera? → z already correlates 0.577 with multi-view
  ground truth; features are scale-normalized so absolute depth is discarded
- Why LSTM, not Transformer? → 196k params fits RPi latency budget;
  accuracy already 72%; topology swap is isolated in build_model()
- Privacy? → video never leaves the device; only recognized word lists go to API
- Why 30 words? → demo-scoped vocabulary; pipeline scales by re-running
  the exact-match label tooling

---

## 발표 리허설 메모
- 8분 = 슬라이드당 평균 ~28s. Slide 9·11·13·14에 시간 집중, 2·3·6은 빠르게.
- "Show that your project works correctly (very important)" → Slide 14 데모가
  사실상 채점 핵심. 데모 영상은 자막(영어)을 포함한다.
