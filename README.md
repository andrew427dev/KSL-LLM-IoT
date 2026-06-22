# KSL-LLM-IoT
## Real-Time Korean Sign Language Recognition with LLM-based Natural Language Generation on Edge Devices

> **IoT Systems Term Project — HUFS Spring 2026**
> Team: Lee Sung-joon (202102467) · Bae Jin-gyu (202001647)

---

## 📌 Project Overview

This project builds a real-time Korean Sign Language (KSL) translation system running entirely on a **Raspberry Pi 4B**. Recognized sign words are passed to a **Large Language Model (Gemini API)** to generate natural Korean sentences, which are then delivered as voice output via TTS and displayed on an I2C LCD.

**Pipeline:**
```
Camera → MediaPipe Hands (2 hands, handedness-aware)
       → 131-dim feature (wrist-relative + intra-hand scale norm
                          + wrist-to-wrist vector + presence flags)
       → LSTM Classifier (TFLite)
       → Word Buffer → Gemini API → TTS + LCD Output
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| Hardware | Raspberry Pi 4B, Pi Camera (v1 / v2 / Module 3), I2C LCD 20×4, GPIO Buzzer, Status LED, Push buttons ×4 (complete + undo + persona ×2) |
| Language | Python **3.11** (mediapipe has no aarch64 wheels for Python 3.13 as of this writing) |
| CV | MediaPipe Hands, OpenCV 4 |
| AI/ML | TensorFlow (PC training) / `tflite-runtime` (RPi inference) |
| LLM | Google Gemini 2.5 Flash via `google-genai` SDK |
| TTS | gTTS + pygame (online) / pyttsx3 (offline fallback) |
| IoT | `RPi.GPIO`, `smbus2`, libcamera / `rpicam-vid` |

---

## 📁 Project Structure

```
KSL-LLM-IoT/
├── data/
│   ├── raw/            # Raw video samples (not committed)
│   ├── landmarks/      # Extracted MediaPipe landmark CSV files
│   └── augmented/      # Augmented dataset
├── model/
│   ├── train.py        # LSTM training
│   ├── augment.py      # Dataset augmentation
│   ├── evaluate.py     # Model evaluation
│   └── ksl_model.tflite  # Compiled TFLite model (not committed)
├── src/
│   ├── main.py             # Entry point, camera backends (OpenCV / rpicam-vid / picamera2)
│   ├── hand_tracker.py     # MediaPipe → 131-dim feature (handedness slots, x-fallback)
│   ├── feature_format.py   # Shared 131-dim layout + normalization (train/serve single source)
│   ├── classifier.py       # TFLite inference (dummy fallback, stale-buffer reset)
│   ├── sentence_builder.py # Word buffer + google-genai (offline fallback, dual-persona cache)
│   ├── button_input.py     # Physical buttons (complete + undo + persona ×2, queue-based)
│   ├── tts_output.py       # gTTS / pyttsx3 voice output (async)
│   └── lcd_display.py      # I2C LCD controller (async queue)
├── config/
│   └── settings.py     # API keys, thresholds, constants
├── docs/
│   ├── wiring_diagram.png
│   └── generate_wiring_diagram.py
├── scripts/                # GPU-server training pipeline (deploy/upload/setup/train/fetch)
├── tools/
│   └── verify_aihub_alignment.py  # AI Hub ↔ MediaPipe axis alignment measurement
├── tests/
├── collect_data.py         # Landmark CSV collection (PC webcam)
├── convert_aihub.py        # AI Hub keypoint JSON → 131-dim CSV (--exact matching)
├── make_dummy_model.py     # Smoke-test dummy TFLite generator
├── requirements.txt        # PC: training + dev
├── requirements-rpi.txt    # RPi: inference only
├── .env.example
├── .gitignore
├── USER_MANUAL.md          # End-user operating guide (Korean)
├── CONTRIBUTING.md
├── CLAUDE.md               # AI agent guidelines for this repo
└── README.md
```

---

## 🗂 KSL Word List (30 words)

| Category | Words |
|----------|-------|
| Pronouns / Response | 나, 당신, 좋다, 싫다, 맞다 |
| Verbs | 가다, 오다, 서다, 자다, 주다 |
| States | 배고프다, 목마르다, 아프다, 피곤하다, 춥다, 덥다, 슬프다, 화나다 |
| Emotion / Intent | 행복, 감사, 부탁, 돕다 |
| Daily Nouns | 밥, 병원, 의사, 엄마, 가족, 친구, 얼마 |
| System | **완료** (transmit trigger sign) |

The word list matches AI Hub sign-language dataset headwords exactly (revised 2026-06-10; see `convert_aihub.py --exact`). Sentence generation is triggered by the **physical complete button** (GPIO5) or the `완료` sign. Silence auto-complete is disabled by default (`SILENCE_TRIGGER_SEC=0`) to avoid conflicting with the undo button; set it to a positive value to re-enable.

---

## 🔌 Hardware Connection

### Components

| Component | Spec | Notes |
|-----------|------|-------|
| Raspberry Pi 4B | RAM 4GB+, 64-bit Raspberry Pi OS (Bookworm or Trixie) | aarch64 |
| Camera | USB webcam (`/dev/video0`) or Pi Camera v1 (OV5647) / v2 (IMX219) / Module 3 (IMX708) via CSI | Auto-detected (`CameraReader`) |
| I2C LCD | 20×4, default address `0x27` | Address override via `LCD_I2C_ADDRESS` in `.env` |
| Active Buzzer | 3.3V or 5V active type | Pin override via `BUZZER_PIN` in `.env` |
| Status LED | Any LED + series resistor (220–330 Ω) | Mirrors the buzzer (visual feedback for deaf users); `LED_PIN` in `.env` |
| Push Buttons ×4 | Momentary tactile switch | Complete + undo + persona ×2, internal pull-up (no external resistor) |
| Speaker | USB-audio type: USB only · Active 3.5mm type: 3.5mm (audio) + USB/external 5V (power) | TTS playback |
| microSD | 32GB+, Class 10 | OS + dataset + model |
| Power | 5V / 3A USB-C | Required when camera + speaker run concurrently |

### Wiring (BCM pin numbers)

| Signal | RPi Pin (BCM) | RPi Pin (Physical) | Component |
|--------|---------------|-------------------|-----------|
| 5V | — | Pin 2 | I2C LCD VCC |
| GND | — | Pin 6 | I2C LCD GND |
| SDA | GPIO2 | Pin 3 | I2C LCD SDA |
| SCL | GPIO3 | Pin 5 | I2C LCD SCL |
| Buzzer signal | **GPIO17** (default `BUZZER_PIN`) | Pin 11 | Active buzzer + |
| Buzzer GND | — | Pin 6 | Active buzzer − (shared GND) |
| LED signal | **GPIO22** (default `LED_PIN`) | Pin 15 | LED anode (+) via 220–330 Ω resistor |
| LED GND | — | Pin 25 | LED cathode (−) — GND below SPI0 SCLK (Pin 23) |
| Complete button | **GPIO5** (`BUTTON_COMPLETE_PIN`) | Pin 29 | Button leg A (leg B → GND Pin 6) |
| Undo button | **GPIO6** (`BUTTON_UNDO_PIN`) | Pin 31 | Button leg A (leg B → GND Pin 6) |
| Persona: polite | **GPIO13** (`BUTTON_PERSONA_POLITE_PIN`) | Pin 33 | Button leg A (leg B → GND Pin 6) |
| Persona: friendly | **GPIO19** (`BUTTON_PERSONA_FRIENDLY_PIN`) | Pin 35 | Button leg A (leg B → GND Pin 6) |
| Camera | — | USB port | USB webcam (`/dev/video0`; CSI Pi Camera also auto-detected) |
| Audio (3.5mm) | — | 3.5mm jack | Active speaker audio in |
| Audio power | — | USB port / external | Active speaker 5V power (USB-audio speaker: USB only) |

Wiring diagram: [`docs/wiring_diagram.png`](docs/wiring_diagram.png) (regenerable with `python docs/generate_wiring_diagram.py`).

### Enabling Interfaces

I2C is disabled by default on a fresh Raspberry Pi OS install. Enable once:

```bash
sudo raspi-config
# → Interface Options → I2C → Enable → Finish → Reboot
```

After reboot, verify hardware detection:

```bash
i2cdetect -y 1                    # I2C devices — expect entry at 0x27 (or configured address)
rpicam-hello --timeout 2000       # Camera — expect libcamera init logs and frame capture
```

---

## ⚙️ Setup

The project uses two distinct environments:

- **PC** — dataset preparation, model training, code development.
- **Raspberry Pi 4B** — real-time inference, hardware integration, demo.

`mediapipe` does not publish aarch64 wheels for Python 3.13. Both environments require **Python 3.11**. On systems where Python 3.11 is not present (e.g., Raspberry Pi OS Trixie ships with 3.13), [`uv`](https://docs.astral.sh/uv/) downloads a prebuilt 3.11 binary without compilation.

### A. PC Environment (training + development)

```bash
# A.1 Clone
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git
cd KSL-LLM-IoT

# A.2 Python 3.11 venv via uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11
source .venv/bin/activate

# A.3 Install dependencies
uv pip install -r requirements.txt

# A.4 Configure env
cp .env.example .env
# Edit .env — GEMINI_API_KEY can stay empty; SentenceBuilder will run in offline fallback.

# A.5 (optional) Smoke-test the pipeline before collecting data or training
python make_dummy_model.py        # Generates a dummy model/ksl_model.tflite
python src/main.py                # GUI mode; press 'q' to quit
```

### B. Raspberry Pi Environment (inference deployment)

```bash
# B.1 System packages
sudo apt update
sudo apt install -y i2c-tools rpicam-apps libcap-dev libcamera-dev \
                    pkg-config python3-dev build-essential

# B.2 Clone (or transfer the repo via pscp — see USER_MANUAL §5.1)
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git ~/Desktop/KSL-LLM-IoT
cd ~/Desktop/KSL-LLM-IoT

# B.3 Python 3.11 venv via uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11
source .venv/bin/activate

# B.4 RPi dependencies (tflite-runtime instead of full TensorFlow)
uv pip install -r requirements-rpi.txt

# B.5 Enable I2C and verify hardware
#   (see "Hardware Connection — Enabling Interfaces" above)

# B.6 Configure env
cp .env.example .env
nano .env
# Required for LLM output: GEMINI_API_KEY=<your key>
# (Key obtainable at https://aistudio.google.com — free tier available, subject to Google's quota)

# B.7 Deploy the trained model
#   Transfer model/ksl_model.tflite from PC to ~/Desktop/KSL-LLM-IoT/model/
#   See USER_MANUAL.md §5.1 for pscp/psftp/WinSCP procedures.

# B.8 Run
python src/main.py                # GUI mode (requires attached display)
KSL_HEADLESS=1 python src/main.py # Headless (SSH / PuTTY) — Ctrl+C to quit
```

### C. Data Collection & Training (PC)

```bash
# C.1 Collect — repeat for each KSL label
python collect_data.py --word 안녕 --samples 100
# In-app: SPACE = start/stop, q = quit
# Output: data/landmarks/<word>/0000.csv, 0001.csv, ...

# C.2 (optional) Augment
python model/augment.py --factor 3

# C.3 Train → model/ksl_model.tflite
python model/train.py

# C.4 Evaluate
python model/evaluate.py
```

After training, transfer `model/ksl_model.tflite` to the Raspberry Pi (Section B.7).

### D. Operation Modes

| `GEMINI_API_KEY` | `model/ksl_model.tflite` | Behavior |
|------------------|-------------------------|----------|
| set | present | Full operation — LLM sentence generation, trained classifier |
| set | absent | LLM works, classifier returns `KSL_LABELS[0]` (dummy mode) |
| empty | present | Classifier works, sentences are space-joined word lists (offline fallback) |
| empty | absent | Full smoke-test mode — pipeline runs end-to-end, no real recognition or generation |

This matrix allows incremental bring-up: hardware → camera → classifier → LLM, each verifiable independently.

---

## 📅 Development Schedule

| Week | Period | Goal |
|------|--------|------|
| Week 1 | May 6–14 | Environment setup, hardware assembly |
| Week 2 | May 14–21 | KSL data collection (30 words × 100 samples) |
| Week 3 | May 21–28 | Model training, TFLite conversion |
| Week 4 | May 28–Jun 1 | Pipeline integration, 1st presentation |
| Week 5 | Jun 1–10 | Gemini API integration, refinement |
| Week 6 | Jun 10–22 | Testing, demo video, final report |

---

## 🎯 Evaluation Targets

| Metric | Target |
|--------|--------|
| Word recognition accuracy | ≥ 85% |
| Inference FPS on RPi 4B | ≥ 20 FPS |
| LLM response latency | ≤ 2 sec |
| End-to-end pipeline delay | ≤ 4 sec |

> **Actual (RPi 4B, develop):** on-device inference is ~6.4 FPS — MediaPipe hand detection is the bottleneck, mitigated by 1-second time-resampling so the recognition window matches the 30 FPS training distribution. The ≥20 FPS target is **not met** and is addressed in Future Work. Recognition accuracy and end-to-end latency are reported in the final report.

---

## 📖 Documentation

- **[USER_MANUAL.md](USER_MANUAL.md)** — End-user operating guide (Korean). Daily operation, data collection, PuTTY file transfer (§5.1), troubleshooting (§6), changelog (§9).
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines.
- **[CLAUDE.md](CLAUDE.md)** — AI agent guidelines and documentation update triggers for this repo.
- **[docs/wiring_diagram.png](docs/wiring_diagram.png)** — Hardware wiring reference.

---

## 📚 References

### Papers

1. Shin, J. et al. (2023). Dynamic Korean Sign Language Recognition Using Pose Estimation-Based and Attention-Based Neural Network. **IEEE Access**, 11, 143501–143513.
2. Miah, A. S. M. et al. (2023). Korean Sign Language Recognition Using Transformer-Based Deep Neural Network. **Applied Sciences**, 13(5), 3029.
3. Sánchez-Vicinaiz, T. J. et al. (2024). MediaPipe + CNN on Raspberry Pi. **Technologies**, 12(8), 124.
4. Fang, S. et al. (2024). SignLLM: Sign Language **Production** LLMs (ICCV 2025). _\*수어 생성(Production) 모델이라 인식용 데이터 출처는 아님. LLM 접근법 참고용._

### Public KSL datasets

| # | 출처 | 규모 | 포맷 | 접근 |
|---|------|------|------|------|
| 1 | [KSL-Guide (KAIST, FG 2021)](https://github.com/ChelseaGH/KSL-Guide) | 121K 영상 / 2,000 문장 + 3,000 단어 | MP4 30FPS + 137 keypoint JSON (양손+몸+얼굴) | AI Hub 가입 + 한국인 한정 |
| 2 | [KSL-77 (Yang et al., 2019)](https://github.com/Yangseung/KSL) | 1,540 영상 / 77 클래스 / 20 signers | RGB 255×255, Optical Flow, MP4 | Dropbox 직접 다운로드, 라벨은 Google Drive |
| 3 | [musaru/KSL (Shin & Miah 2023 코드)](https://github.com/musaru/KSL) | KSL-77 기반 GCN 구현 | PyTorch + STGCN.ipynb, 47 pose landmark | 코드 공개 |
| 4 | [AIRC-KETI/GKSL-dataset](https://github.com/AIRC-KETI/GKSL-dataset) | Gloss 레벨 KSL | (확인 필요) | LICENSE.md 별도 |
| 5 | [KETI 응급 KSL (arXiv:1811.11436)](https://arxiv.org/abs/1811.11436) | 14,672 영상 / 419 단어 + 105 문장 | RGB + OpenPose | 비공개 가능성, 논문 contact 필요 |

> 본 프로젝트의 30단어 인식 모델은 **AI Hub 수어 영상 데이터셋의 3D 키포인트**(시연자 16명)를 `convert_aihub.py --exact`로 프로젝트 131차원 포맷으로 변환해 학습한다. 좌표계 정렬(축 부호·등방 보정)은 `tools/verify_aihub_alignment.py`로 동일 영상의 MP4와 키포인트를 프레임 단위로 비교해 실측 검증했다. `collect_data.py`를 통한 **자체 수집**은 데모 화자 도메인 보강용으로 혼합 학습한다(future work). 위 표의 외부 데이터셋은 라벨 매핑·키포인트 정의(MediaPipe 21점 vs OpenPose/137점) 검토용 참고 목록이다.

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.
