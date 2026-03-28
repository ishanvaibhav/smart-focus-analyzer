# 🧠 Smart Focus & Distraction Analyzer

> An IoT + ML project that uses your webcam to detect whether you're **focused**, **distracted**, or **drowsy** — with a live dashboard, session history chart, and focus score.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Features

| Feature | Description |
|---|---|
| 🎯 Face Detection | Detects if you're looking at the screen |
| 👁️ Eye Detection | Identifies drowsiness from closed eyes |
| 📊 Focus Score | Live % score updated each capture |
| 📈 Session Chart | Plotly graph of your focus over time |
| 🎤 Noise Indicator | Environment noise level (manual slider) |
| ☁️ Cloud Ready | Deploys to Streamlit Cloud for free |

---

## 🚀 Quick Start (Run Locally)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-focus-analyzer.git
cd smart-focus-analyzer
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501` automatically.

---

## 📂 Project Structure

```
smart-focus-analyzer/
│
├── app.py               ← Main Streamlit app (UI + logic)
├── detector.py          ← Face + eye detection (OpenCV)
├── requirements.txt     ← All Python dependencies
├── .gitignore           ← Files Git should ignore
├── .streamlit/
│   └── config.toml      ← Dark theme + server settings
└── README.md            ← This file
```

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. **Push this project to GitHub** (see section below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/smart-focus-analyzer`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy** — done! You get a free public URL.

> ⚠️ Note: Streamlit Cloud supports `st.camera_input()` in modern browsers.
> Make sure to **allow camera access** when the browser prompts you.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** — face and eye detection via Haar Cascades
- **Streamlit** — web UI framework
- **Plotly** — interactive charts
- **Pandas** — session data handling
- **Pillow** — image processing

---

## 🧠 How It Works

```
Webcam Image
     │
     ▼
FaceAttentionDetector (detector.py)
     │
     ├── No face detected?        → DISTRACTED
     ├── Face off-centre?         → DISTRACTED  
     ├── Eyes closed too long?    → DROWSY
     └── All checks pass?         → FOCUSED
     │
     ▼
Focus Score = (Focused Frames / Total Frames) × 100
     │
     ▼
Streamlit Dashboard (app.py)
  • Status badge (green / red / orange)
  • Gauge chart
  • Session history line chart
```

---

## 📈 Roadmap (Future Upgrades)

- [ ] MediaPipe face mesh for higher accuracy
- [ ] ML model trained on custom data (replaces rule-based logic)
- [ ] Phone/app usage detection
- [ ] Pomodoro AI (smart break timer)
- [ ] Email/Slack daily focus report
- [ ] AWS IoT Core integration for cloud logging

---

## 💼 Interview Description

> *"Built an IoT + ML-based attention monitoring system using computer vision and behavioural analytics to generate real-time focus insights and adaptive productivity recommendations — deployed as a full-stack web app on Streamlit Cloud."*

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Your Name**  
[GitHub](https://github.com/YOUR_USERNAME) • [LinkedIn](https://linkedin.com/in/YOUR_USERNAME)
