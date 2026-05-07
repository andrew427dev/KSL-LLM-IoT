# KSL-LLM-IoT
## Real-Time Korean Sign Language Recognition with LLM-based Natural Language Generation on Edge Devices

> **IoT Systems Term Project — HUFS Spring 2026**
> Team: Lee Sung-joon (202102467) · Bae Jin-gyu (202001647)

---

## 📌 Project Overview

This project builds a real-time Korean Sign Language (KSL) translation system running entirely on a **Raspberry Pi 4B**. Recognized sign words are passed to a **Large Language Model (Gemini API)** to generate natural Korean sentences, which are then delivered as voice output via TTS and displayed on an I2C LCD.

**Pipeline:**
```
Camera → MediaPipe Hands → LSTM Classifier (TFLite)
       → Word Buffer → Gemini API → TTS + LCD Output
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| Hardware | Raspberry Pi 4B, Pi Camera v2, I2C LCD 20x4, GPIO Buzzer |
| Language | Python 3.11 |
| CV | MediaPipe Hands, OpenCV 4 |
| AI/ML | TensorFlow / TFLite (LSTM) |
| LLM | Google Gemini 2.5 Flash API |
| TTS | gTTS / pyttsx3 |
| IoT | RPi.GPIO, smbus2 |

---

## 📁 Project Structure

```
KSL-LLM-IoT/
├── data/
│   ├── raw/            # Raw sign language video samples (not committed)
│   ├── landmarks/      # Extracted MediaPipe landmark CSV files
│   └── augmented/      # Augmented dataset
├── model/
│   ├── train.py        # LSTM model training script
│   ├── evaluate.py     # Model evaluation
│   └── ksl_model.tflite  # Compiled TFLite model
├── src/
│   ├── main.py             # Main entry point
│   ├── hand_tracker.py     # MediaPipe landmark extraction
│   ├── classifier.py       # TFLite inference
│   ├── sentence_builder.py # Word buffer + Gemini API
│   ├── tts_output.py       # gTTS / pyttsx3 voice output
│   └── lcd_display.py      # I2C LCD controller
├── config/
│   └── settings.py     # API keys, thresholds, constants
├── tests/
│   └── test_classifier.py
├── docs/
│   └── wiring_diagram.png
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🗂 KSL Word List (30 words)

| Category | Words |
|----------|-------|
| Greetings | 안녕, 감사합니다, 미안합니다, 반갑습니다 |
| Pronouns | 나/저, 당신/너, 우리 |
| Yes/No | 예/네, 아니오, 좋다, 싫다, 맞다 |
| Basic Verbs | 먹다, 마시다, 가다, 오다, 앉다, 서다, 자다 |
| States | 배고프다, 목마르다, 아프다, 피곤하다, 행복하다 |
| Requests | 도와주세요, 주세요, 기다리세요 |
| Others | 화장실, 얼마예요, 완료(trigger) |

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/andrew427dev/KSL-LLM-IoT.git
cd KSL-LLM-IoT
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 5. Collect data (Week 2)
```bash
python collect_data.py --word 안녕 --samples 100
```

### 6. Augment dataset (Week 3)
```bash
# data/landmarks/ → data/augmented/ (원본 1개당 증강 3개 생성)
python model/augment.py --factor 3
```

### 7. Train model (Week 3)
```bash
python model/train.py
```

### 8. Run the system
```bash
python src/main.py
```

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

---

## 📚 References

1. Shin, J. et al. (2023). Dynamic Korean Sign Language Recognition. **IEEE Access**, 11, 143501–143513.
2. Miah, A. S. M. et al. (2023). KSL Recognition Using Transformer. **Applied Sciences**, 13(5), 3029.
3. Fang, S. et al. (2024). SignLLM: Sign Language Production LLMs. **(ICCV 2025)**
4. Sánchez-Vicinaiz, T. J. et al. (2024). MediaPipe + CNN on Raspberry Pi. **Technologies**, 12(8), 124.
5. KSL-Guide Dataset (IEEE FG 2021): https://github.com/ChelseaGH/KSL-Guide

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.
