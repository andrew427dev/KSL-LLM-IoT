# KSL-LLM-IoT — Presentation Slides (English, ≥15)
# 8 min talk + 2 min Q&A. Source for the .pptx deliverable.

---

## Slide 1 — Title
**Real-Time Korean Sign Language Translation on a Raspberry Pi**
*with LLM-based Natural-Language Sentence Generation*
IoT Systems, Spring 2026 · Lee Sungjoon (202102467), Bae Jingyu (202001647)

## Slide 2 — The Problem
- ~400,000 deaf people in Korea; KSL is their primary language
- Very few hearing people understand KSL → daily communication barrier
- Prior work mostly stops at **word-level** recognition
- Goal: end-to-end **words → natural sentence → speech**, on a low-cost **edge** device

## Slide 3 — Concept (One-Line Pipeline)
Camera → MediaPipe (2 hands) → 131-dim feature → LSTM (TFLite) → word buffer
→ [button | `완료` sign | 3 s silence] → Gemini 2.5 Flash → TTS (Korean) + LCD (English)
- Perception is **local** (privacy, latency); only sentence generation calls the cloud

## Slide 4 — System Architecture
- Presentation: LCD + TTS
- Application: sentence buffer + LLM controller (persona)
- AI/ML: MediaPipe → LSTM (TFLite)
- Hardware: RPi 4B, camera, GPIO buttons, buzzer

## Slide 5 — Two-Hand 131-Dim Feature
- KSL is two-handed → single-hand (63-d) loses meaning
- 131 = [L 21×3 | R 21×3 | wrist-to-wrist 3 | presence 2]
- Per-hand **intra-hand scale** normalization → invariant to distance, hand size, units

## Slide 6 — Train/Serve Consistency by Construction
- Inference and dataset conversion share **one module** (`feature_format.py`)
- Same code builds the vector in both paths → preprocessing skew is **structurally impossible**

## Slide 7 — Empirical Axis Alignment (Key Contribution)
- AI-Hub keypoints (meters, non-mirrored) vs MediaPipe ([0,1], mirrored)
- Compared the same clip frame-by-frame (725 pairs)
- Fix = **x-axis flip**; correlation x=0.954, y=0.982, z=0.577
- Without it: x=−0.954 → a left-right-flipped model. Plus (w,h,w) isotropy correction

## Slide 8 — Temporal Resampling (Low-FPS Robustness)
- On-device MediaPipe ≈ 8–9 FPS → a 30-frame buffer stretches 1 s into ~4 s
- Keep (timestamp, vector); **resample last 1.0 s into 30 points** → matches 30-FPS training window

## Slide 9 — Leakage-Corrected Evaluation
- Random frame split gave a misleading **1.0000** (windows/augments leaked)
- Switched to **held-out-signer** split; held-out augments excluded from train *and* test
- Only unseen-signer accuracy is used for judgment

## Slide 10 — LLM Sentence Generation + Persona
- Word buffer → Gemini 2.5 Flash, persona system prompt (polite/friendly/brief)
- Async worker (camera never blocks); offline fallback = word concatenation

## Slide 11 — Hardware & Accessibility
- USB webcam, I2C LCD 20×4, active buzzer, 4 push buttons (complete + 3 personas)
- Internal pull-ups to GND (no external resistors)
- **Beep 1/2/3 times** = non-visual persona confirmation for deaf/HoH users

## Slide 12 — Software Stack
- CV: MediaPipe Hands (<0.10.30), OpenCV
- Edge: tflite-runtime, Python 3.11 (uv); camera via rpicam-vid on RPi
- Train: TensorFlow 2.15.1 (GPU server); LLM: google-genai

## Slide 13 — Engineering Pitfalls Solved
- LabelEncoder Unicode sort broke index↔label → original-order mapping + test guard
- TF ≥2.16 LSTM-TFLite fails / float16 OOM → TF 2.15.1, batch-1 convert, float32 (740 KB)
- Silent retrain on stale code → `chain_train.sh` fetch + reset + code-marker check

## Slide 14 — Results
- Deployed TFLite, held-out 2 unseen signers: **0.94** accuracy
- On-device (USB webcam): **27/30** words reliable
- Model 740 KB; inference 0.41 ms (server CPU); end-to-end 6.4 FPS

## Slide 15 — Evaluation Integrity (Keras vs TFLite)
- Same 2,595 inputs: Keras 0.71 (CPU=GPU) vs deployed **TFLite 0.94**
- Disagree on 30%; on those, TFLite correct 650 vs Keras 59
- RPi runs the TFLite → 0.94 is the faithful deployed number (reported with caveats)

## Slide 16 — Limitations (Honest)
- Held-out = 2 signers → wide confidence interval
- `밥/배고프다/주다` still confusable on-device
- ≥20 FPS target not met (MediaPipe bottleneck) → Future Work

## Slide 17 — Contributions
- Reusable dataset↔runtime alignment verification (`verify_aihub_alignment.py`)
- Single-source preprocessing (no train/serve skew)
- Leakage-corrected signer-level evaluation; honest accuracy trajectory
- Accessibility-first control; automated cloud-train → edge-deploy

## Slide 18 — Future Work & Demo
- Continuous signing (SEN), vocabulary scaling, pose keypoints (position-dependent signs)
- On-device LLM (offline), higher-FPS path
- **Live / recorded demo** → Q&A
