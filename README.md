# Smart Focus & Distraction Analyzer

An IoT + ML project that uses your webcam to detect whether you're focused, distracted, or drowsy, with a live dashboard, session history chart, and focus score.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

| Feature | Description |
|---|---|
| Live webcam streaming | Uses `streamlit-webrtc` for a continuous browser camera feed |
| Face detection | Detects whether your face is present and centered |
| Eye detection | Flags possible drowsiness when eyes stay closed |
| Focus score | Tracks the percentage of focused frames in real time |
| Session history | Shows a rolling chart of focus states and score |
| Noise indicator | Lets you simulate environment noise from the sidebar |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ishanvaibhav/smart-focus-analyzer.git
cd smart-focus-analyzer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

If port `8501` is already in use, run:

```bash
streamlit run app.py --server.port 8503
```

Then open the local URL shown in the terminal.

### 5. Start the live stream

1. Open the app in your browser.
2. Click `Start` in the `Live Camera Stream` section.
3. Allow camera access when your browser prompts you.
4. Watch the dashboard update automatically while the stream stays live.

---

## Troubleshooting

- If the stream does not start, make sure you clicked `Start` and allowed camera access in the browser.
- If you see a missing dependency message, run `pip install -r requirements.txt` again in the same Python environment you use to launch Streamlit.
- If `8501` is busy, launch with `--server.port 8503` or another open port.
- If you are not on `localhost`, use a browser and host setup that allows webcam access.
- For remote deployment, add TURN credentials in Streamlit secrets with `turn_server_url`, `turn_username`, and `turn_password` for a more reliable WebRTC connection.

---

## Project Structure

```text
smart-focus-analyzer/
|
|-- app.py                Main Streamlit app (UI + live stream + dashboard)
|-- detector.py           Face + eye detection logic using OpenCV
|-- requirements.txt      Python dependencies
|-- .gitignore            Git ignore rules
|-- .streamlit/
|   |-- config.toml       Streamlit theme and server settings
|-- README.md             Project documentation
```

---

## How It Works

```text
Browser Webcam
     |
     v
streamlit-webrtc live stream
     |
     v
FaceAttentionDetector (detector.py)
     |
     |-- No face detected?      -> DISTRACTED
     |-- Face off-centre?       -> DISTRACTED
     |-- Eyes closed too long?  -> DROWSY
     |-- Otherwise              -> FOCUSED
     |
     v
Focus Score = (Focused Frames / Total Frames) * 100
     |
     v
Streamlit dashboard
  - Status badge
  - Focus gauge
  - Session history chart
```

---

## Deploy to Streamlit Cloud

1. Push the project to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Create a new app.
4. Select:
   - Repository: `ishanvaibhav/smart-focus-analyzer`
   - Branch: `main`
   - Main file path: `app.py`
5. Deploy and allow camera access in the browser.

Note: this app uses `streamlit-webrtc`, so live camera access depends on browser permissions and the runtime environment.
Optional: add TURN credentials in Streamlit secrets using `turn_server_url`, `turn_username`, and `turn_password` if the remote stream does not connect reliably.

---

## Tech Stack

- Python
- Streamlit
- streamlit-webrtc
- OpenCV
- Plotly
- Pandas

---

## Roadmap

- [ ] MediaPipe face mesh for higher accuracy
- [ ] ML model trained on custom data
- [ ] Phone or app usage detection
- [ ] Pomodoro timer integration
- [ ] Daily focus reports
- [ ] Cloud logging

---

## License

MIT License
