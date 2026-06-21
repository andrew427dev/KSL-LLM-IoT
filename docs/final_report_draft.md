# Real-Time Korean Sign Language Translation on a Raspberry Pi
### with LLM-based Natural-Language Sentence Generation

**Course:** Internet of Things (IoT) Systems — Spring 2026, HUFS
**Team:** Lee Sungjoon (202102467), Bae Jingyu (202001647)
**Submission:** 2026-06-22

> Draft report (English). Figures are referenced as `Fig.x`; final figures are produced from
> `docs/wiring_diagram.png`, `model/confusion_matrix.png`, and demo screenshots.
> Numbers below are measured in this repository unless marked *not yet measured*.

---

## Abstract

This system, **KSL-LLM-IoT**, is a real-time Korean
Sign Language (KSL) translator that runs on a Raspberry Pi 4B edge device. A camera captures
two-hand signing; MediaPipe extracts 3D hand landmarks; a TensorFlow-Lite LSTM classifies 30 KSL
words; and a large language model (Gemini 2.5 Flash) composes the recognized word sequence into a
natural Korean sentence, delivered through a speaker (TTS) and an I2C LCD. Physical buttons provide
an accessible, latency-free control interface. On a held-out set of two unseen signers the deployed
TFLite model reaches **0.72** word-classification accuracy; on the physical device 27 of 30 words are
recognized reliably. The project's main engineering contributions are (i) an *empirically verified*
coordinate-system alignment between a public keypoint dataset and the runtime camera, (ii) a
single-source feature representation that structurally eliminates train/serve skew, and (iii) a
leakage-corrected, signer-level evaluation protocol.

---

## 1. Project Idea

**Problem.** Roughly 400,000 deaf people in Korea use KSL as their primary language, yet very few
hearing people understand it, creating a daily communication barrier that today depends on scarce
human interpreters. Most prior KSL work stops at word-level classification; an end-to-end pipeline
that produces *natural sentences and speech on an edge device* is rare.

**Concept.** A low-cost Raspberry Pi performs all perception locally (privacy, low latency) and calls
a cloud LLM only for the final sentence-generation step. The pipeline is:

> Camera → MediaPipe (2 hands) → 131-dim feature → LSTM (TFLite) → word buffer →
> [complete button | `완료` sign] → Gemini 2.5 Flash → TTS (Korean) + LCD (English).

**IoT characteristics.** Sensors/actuators (camera, GPIO buttons, buzzer, status LED) → on-device inference →
outputs (LCD, speaker); plus a cloud-train → edge-deploy lifecycle.

**Differentiators.** (1) Natural LLM sentences rather than a word list; (2) user-selectable sentence
*persona* (polite/friendly); (3) accessibility-first control (physical buttons + synchronized
buzzer/LED feedback — the LED gives visual confirmation for deaf/hard-of-hearing users); (4) a *measured* train/serve coordinate
alignment between the AI-Hub dataset and the runtime camera.

---

## 2. System Design & Methodology

### 2.1 Pipeline (Fig.1)
`Camera → MediaPipe(2 hands) → 131-dim feature → LSTM(TFLite) → word buffer → trigger → Gemini → TTS + LCD`

### 2.2 Two-hand 131-dim representation (Fig.2)
KSL is a two-handed language, so a single-hand (63-dim) input cannot preserve meaning. The feature is

`131 = [ LEFT 21×3 | RIGHT 21×3 | wrist-to-wrist vector 3 | presence flags 2 ]`.

Each hand is normalized by its wrist-relative coordinates divided by an **intra-hand scale**
(‖landmark9 − landmark0‖), making it invariant to camera distance, hand size, and coordinate units.
A missing hand is zero-filled with its presence flag cleared.

### 2.3 Train/serve consistency by construction
Inference (`hand_tracker.py`) and dataset conversion (`convert_aihub.py`) share a single module,
`src/feature_format.py`. Because the exact same code assembles the 131-dim vector in both paths,
preprocessing-level train/serve skew is structurally impossible.

### 2.4 Empirical axis alignment (Fig.3)
AI-Hub provides multi-view 3D-reconstructed keypoints (meters, non-mirrored); MediaPipe provides
normalized [0,1] coordinates from a mirrored selfie image. We compared the MP4 (runtime path) and the
keypoints of the *same clip* frame-by-frame (725 pairs):

- required transform = **x-axis sign flip** (`AIHUB_AXIS_SIGNS = (-1, 1, 1)`);
- correlation after transform: **x = 0.954, y = 0.982, z = 0.577** (without it, x = −0.954 → a
  left-right-flipped model);
- z = 0.577 reflects the limits of monocular depth but is a directionally consistent auxiliary signal
  (no ToF camera needed);
- an additional **(w, h, w) isotropy correction** (gap "G6") aligns MediaPipe's per-axis normalization
  (x ÷ width, y ÷ height) to the dataset's isotropic metric coordinates.

### 2.5 Classifier and temporal normalization
2×LSTM (128, 64) + Dense, input (30, 131), 30 classes, exported to a **740 KB float32 TFLite** model.
Because on-device MediaPipe runs at only ~8–9 FPS, a fixed 30-frame buffer would stretch a 1-second
sign into a ~4-second window. The classifier instead keeps a (timestamp, vector) buffer and **linearly
resamples the most recent 1.0 s into 30 points**, restoring the training-time temporal window.

### 2.6 Leakage-corrected, signer-level evaluation
An early model evaluated with a random frame-level split reported a misleading 1.0000 test accuracy
because sliding windows and augmentations of the same clip leaked across train/test. We switched to a
**held-out-signer split** (`model/data_split.py`, `holdout.json`): two signers (REAL01, REAL02) are
held out entirely, and their augmented variants are excluded from both train and test. The held-out
accuracy of unseen signers is the only number used for judgment.

### 2.7 LLM sentence generation
The word buffer is sent to Gemini 2.5 Flash with a persona-specific system prompt; the call runs in an
async worker so the camera loop never blocks, and falls back to a plain word concatenation when offline.

---

## 3. Hardware & Software Details

### 3.1 Hardware (Fig.4 — `docs/wiring_diagram.png`)

| Component | Interface | Pin |
|---|---|---|
| USB webcam (`/dev/video0`, used in the demo) — Pi Camera v1/CSI also supported | USB / CSI | — |
| I2C LCD 20×4 (0x27) | I2C | GPIO2/3 |
| Active buzzer | GPIO out | GPIO17 |
| Status LED (mirrors buzzer, via resistor) | GPIO out | GPIO22 |
| Push buttons ×4 (complete + undo + persona×2) | GPIO in (internal pull-up) | GPIO5/6/13/19 ↔ GND |
| Speaker | 3.5 mm / USB (separate power) | — |

Buttons use internal pull-ups switching to GND (~66 µA, no external resistor): complete (GPIO5),
undo (GPIO6), and two persona buttons (polite GPIO13, friendly GPIO19). The buzzer beeps
1/2 times to confirm the selected persona, and a status LED (GPIO22) blinks in sync with every
beep so deaf/HoH users receive the same cue visually. Sentence completion is signalled by a single
*long* beep+LED (distinct from the short per-word cue). The undo button is context-sensitive: while
words are being entered it removes the last buffered word (short triple beep); after a sentence is
produced it re-generates only the current persona's sentence (long beep) to recover from a wrong
reading. Because one completion call generates both personas at once and caches them, switching
persona or re-pressing complete replays the cached sentence with zero extra API calls.

### 3.2 Software stack

| Layer | Technology | Notes |
|---|---|---|
| Computer vision | MediaPipe Hands (<0.10.30), OpenCV | legacy solutions API |
| Edge inference | tflite-runtime | RPi, Python 3.11 via `uv` |
| Training | TensorFlow 2.15.1 (GPU) | cloud server (RTX 4000 Ada) |
| LLM | google-genai (Gemini 2.5 Flash) | persona system prompt, async |
| OS / IoT | Raspberry Pi OS Trixie, RPi.GPIO (polling), smbus2 | rpicam-vid camera backend |

**Implementation notes (code).** `feature_format.py` (shared 131-dim assembly); `button_input.py`
(polling edge detection — Trixie's libgpiod rejects sysfs event-detect); `sentence_builder.py` (async
worker + persona + guaranteed in-flight release); cloud-train→edge-deploy scripts (`scripts/`); a
numpy-only CI that runs leakage/label-encoding regression guards.

### 3.3 Engineering pitfalls solved (highlights)
- **LabelEncoder bug:** scikit-learn sorted labels by Unicode, breaking the index↔label mapping → fixed
  with an original-order `LABEL_TO_IDX` plus a smoke-test guard.
- **TFLite conversion:** TF ≥2.16 cannot convert the LSTM (MLIR bug) and float16 quantization OOMs →
  pinned TF 2.15.1, batch-size-1 conversion, float32 (model is only 740 KB).
- **Silent training failure:** an `exit 0` retrain had silently trained on stale code (a dirty server
  tree made `git pull` fail without error) → `chain_train.sh` uses fetch + `reset --hard` + a code-marker
  grep before training.

---

## 4. Results & Contributions

### 4.1 Quantitative results

| Item | Value |
|---|---|
| Pipeline validation (example 18 classes, 3,540 samples) | TFLite accuracy 0.98 |
| **Deployed model** (30 classes, 16 AI-Hub signers) — held-out (REAL01/02, 2,595 seq) | **TFLite accuracy 0.72** (0.7168) |
| Physical device (USB webcam, `diagnose_live`) | 27/30 words reliable, confidence 0.3–0.9 |
| TFLite inference latency | 0.41 ms (server CPU) / *not yet measured* (RPi) |
| End-to-end FPS on RPi 4B | 6.4 (MediaPipe bottleneck; **target ≥20 not met** — see Future Work) |
| Model size | 740 KB |
| Sentence latency (Gemini) | *not yet measured* (target ≤ 4 s) |

### 4.2 Evaluation integrity: a corrected TFLite state-carryover artifact
An earlier evaluation reported 0.94, while the Keras checkpoint scored 0.7129 on the *same* 2,595
held-out inputs. We traced the gap to a bug in the evaluation harness, not the model: the TFLite
interpreter was reused across samples and its fused `UnidirectionalSequenceLSTM` cell/hidden state
was **not reset between `invoke()` calls**. Because the held-out set is stored in contiguous per-label
blocks, each sample inherited residual state from a *same-label* neighbor, inflating accuracy to 0.94.
Resetting state on every inference (`reset_all_variables()`) makes TFLite reproduce the Keras number
**exactly (0.7168, bit-for-bit)**; shuffling the same leaked evaluation collapses it to 0.45, confirming
the artifact. We fixed both the evaluation script (`model/evaluate.py`) and the on-device inference path
(`src/classifier.py`) to reset state per window, and **report 0.72 (0.7168) as the deployed model's
held-out accuracy** on two unseen signers. (The confusion matrix produced by the buggy run is invalid
and is being regenerated with the corrected harness.)

### 4.3 Honest limitations
Accuracy is reported on two held-out signers (wide confidence interval), recognition of `밥/배고프다/주다`
remains confusable on-device, and the FPS target is not met. These are discussed rather than hidden.

### 4.4 Contributions
1. A reusable methodology and tool (`tools/verify_aihub_alignment.py`) for *empirically verifying*
   coordinate alignment between a public keypoint dataset and a runtime extractor.
2. Single-source feature preprocessing that structurally prevents train/serve skew.
3. A leakage-corrected, signer-level evaluation that documents the honest accuracy trajectory
   (1.0000 leaked → 0.36 single-signer → 0.72 with 16 signers).
4. Accessibility-first interface: deterministic physical buttons replace recognition-latency triggers;
   feedback is both audible (buzzer) and visual (a buzzer-synchronized LED) for deaf/HoH users, with a
   distinct long cue for sentence completion and a replay-last-sentence action.
5. A fully automated cloud-train → edge-deploy pipeline.

---

## 5. Future Work

- Continuous-signing recognition using the AI-Hub sentence (SEN) dataset.
- Vocabulary scaling (30 → hundreds); the label↔headword pipeline already generalizes.
- Add upper-body **pose keypoints** to separate position-dependent signs (`나` vs `당신`, gap "G7").
- On-device lightweight LLM (e.g., Gemma) for fully offline operation.
- Higher-FPS path (MediaPipe Tasks API) to close the ≥20 FPS gap.
- Korean-capable graphic LCD or a mobile companion app.

---

## References
1. Shin, J. et al. (2023). Dynamic KSL Recognition. *IEEE Access* 11.
2. Miah, A. S. M. et al. (2023). KSL Recognition Using a Transformer DNN. *Applied Sciences* 13(5).
3. Sánchez-Vicinaiz, T. J. et al. (2024). MediaPipe + CNN on Raspberry Pi. *Technologies* 12(8).
4. AI-Hub Korean Sign Language video dataset (keypoints).
