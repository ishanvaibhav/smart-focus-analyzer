"""
app.py — Smart Focus & Distraction Analyzer
============================================
Streamlit app that works BOTH locally and on Streamlit Cloud.

HOW IT WORKS:
  • Takes a photo from your webcam using st.camera_input()
  • Analyzes the image using OpenCV (face + eye detection)
  • Tracks your focus score and session history
  • Shows a live dashboard with charts

RUN LOCALLY:
  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import plotly.graph_objects as go
from detector import FaceAttentionDetector

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Focus Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — makes it look professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .status-box {
        padding: 18px 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .focused    { background: #1a3a1a; color: #39ff14; border: 2px solid #39ff14; }
    .distracted { background: #3a1a1a; color: #ff4444; border: 2px solid #ff4444; }
    .drowsy     { background: #3a2a0a; color: #ffaa00; border: 2px solid #ffaa00; }
    .score-big  { font-size: 52px; font-weight: 900; text-align: center; }
    .tip-box    { background: #1a1a2e; border-left: 4px solid #7c83fd;
                  padding: 12px 16px; border-radius: 8px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE — persists data across reruns
# ─────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "session_start" not in st.session_state: st.session_state.session_start = time.time()
if "total_frames"  not in st.session_state: st.session_state.total_frames  = 0
if "focused_count" not in st.session_state: st.session_state.focused_count = 0
if "detector"      not in st.session_state: st.session_state.detector      = FaceAttentionDetector()

detector = st.session_state.detector

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("🧠 Focus Analyzer")
    st.markdown("---")

    st.subheader("⚙️ Settings")
    center_tolerance = st.slider(
        "Face Centre Tolerance", 0.10, 0.40, 0.20, 0.05,
        help="How much can your face deviate from centre before you're 'distracted'?"
    )
    detector.CENTER_TOLERANCE = center_tolerance

    eye_threshold = st.slider(
        "Drowsy Eye Frames", 5, 30, 15, 1,
        help="How many consecutive frames with eyes closed = drowsy?"
    )
    detector.EYE_CLOSED_THRESHOLD = eye_threshold

    st.markdown("---")
    st.subheader("🔢 Session Stats")
    elapsed = int(time.time() - st.session_state.session_start)
    mins, secs = divmod(elapsed, 60)
    st.metric("⏱ Session Time", f"{mins:02d}:{secs:02d}")
    st.metric("📸 Frames Analysed", st.session_state.total_frames)

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.history       = []
        st.session_state.session_start = time.time()
        st.session_state.total_frames  = 0
        st.session_state.focused_count = 0
        st.success("Session reset!")

    st.markdown("---")
    st.caption("Built with OpenCV + Streamlit")
    st.caption("🔗 [View on GitHub](https://github.com)")

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.title("🧠 Smart Focus & Distraction Analyzer")
st.markdown("Point your camera at your **face**, then click **Take Photo** to analyse your focus state.")
st.markdown("---")

col_cam, col_results = st.columns([1.2, 1], gap="large")

# ── LEFT: Camera Input ─────────────────────────────────────────
with col_cam:
    st.subheader("📷 Camera Feed")
    camera_image = st.camera_input(
        label="Click 'Take Photo' to capture and analyse",
        label_visibility="collapsed"
    )

    # Noise Level — manual slider (mic not available on cloud)
    st.markdown("---")
    st.subheader("🎤 Environment Noise Level")
    st.caption("Adjust this to match your surroundings (mic access not available on cloud)")
    noise_val = st.slider("Noise Level", 0, 100, 20)
    if noise_val < 30:
        noise_label = "🟢 Quiet"
        noise_color = "green"
    elif noise_val < 65:
        noise_label = "🟡 Moderate"
        noise_color = "orange"
    else:
        noise_label = "🔴 Loud"
        noise_color = "red"

    st.markdown(
        f"<div style='font-size:18px; color:{noise_color}; font-weight:bold;'>{noise_label}</div>",
        unsafe_allow_html=True
    )

# ── RIGHT: Results ─────────────────────────────────────────────
with col_results:
    st.subheader("📊 Analysis Results")

    # Compute focus score
    total  = st.session_state.total_frames
    scored = st.session_state.focused_count
    score  = (scored / total * 100) if total > 0 else 0.0

    # ── Score Gauge ──────────────────────────────────────────
    gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": "Focus Score %", "font": {"size": 16}},
        number= {"suffix": "%", "font": {"size": 40}},
        gauge = {
            "axis"      : {"range": [0, 100], "tickwidth": 1},
            "bar"       : {"color": "#39ff14"},
            "bgcolor"   : "#1e1e2e",
            "bordercolor": "gray",
            "steps": [
                {"range": [0,  40], "color": "#3a1a1a"},
                {"range": [40, 70], "color": "#3a2a0a"},
                {"range": [70,100], "color": "#1a3a1a"},
            ],
            "threshold": {
                "line" : {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": 70
            }
        }
    ))
    gauge.update_layout(
        height=240,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="#0f1117",
        font=dict(color="white")
    )
    st.plotly_chart(gauge, use_container_width=True)

    # ── Status + Result ──────────────────────────────────────
    if camera_image is not None:
        # Convert uploaded image → numpy array for OpenCV
        pil_img = Image.open(camera_image).convert("RGB")
        frame   = np.array(pil_img)
        frame   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        result  = detector.analyze(frame, draw=True, show_eyes=True)
        status  = result["status"]
        reason  = result["reason"]
        ann_frame = result["frame"]

        # Apply noise override
        if noise_val >= 65 and status == "focused":
            status = "at_risk"
            reason = "Focused but loud environment"

        # Update session counters
        st.session_state.total_frames += 1
        if status == "focused":
            st.session_state.focused_count += 1

        # Save to history
        st.session_state.history.append({
            "time" : time.strftime("%H:%M:%S"),
            "status": status,
            "score" : round((st.session_state.focused_count /
                             st.session_state.total_frames) * 100, 1),
            "noise" : noise_label
        })

        # Status box
        STATUS_CONFIG = {
            "focused"   : ("focused",    "✅ FOCUSED",     "You're in the zone! Keep it up."),
            "distracted": ("distracted", "❌ DISTRACTED",  "Look directly at the screen."),
            "drowsy"    : ("drowsy",     "😴 DROWSY",      "Take a short break or splash water on your face."),
            "at_risk"   : ("distracted", "⚠️ AT RISK",     "Noisy environment may hurt focus."),
        }
        css_class, label, tip = STATUS_CONFIG.get(
            status, ("distracted", "❓ UNKNOWN", "Try again."))

        st.markdown(
            f'<div class="status-box {css_class}">{label}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Reason:** {reason}")
        st.markdown(
            f'<div class="tip-box">💡 <b>Tip:</b> {tip}</div>',
            unsafe_allow_html=True
        )

        # Show annotated frame below camera
        with col_cam:
            st.markdown("**Annotated Frame:**")
            ann_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True)
    else:
        st.info("👆 Take a photo above to see your focus analysis here.")

# ─────────────────────────────────────────────
# SESSION HISTORY CHART
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Session Focus History")

if len(st.session_state.history) >= 2:
    df = pd.DataFrame(st.session_state.history)

    STATUS_SCORE = {"focused": 100, "at_risk": 60, "distracted": 20, "drowsy": 10}
    df["value"] = df["status"].map(STATUS_SCORE)

    STATUS_COLORS_MAP = {
        "focused"   : "#39ff14",
        "distracted": "#ff4444",
        "drowsy"    : "#ffaa00",
        "at_risk"   : "#ff8800",
    }

    fig = go.Figure()

    # Plot coloured dots per status
    for s, color in STATUS_COLORS_MAP.items():
        mask = df["status"] == s
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask]["time"],
                y=df[mask]["value"],
                mode="markers",
                marker=dict(color=color, size=12),
                name=s.capitalize()
            ))

    # Rolling focus score line
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["score"],
        mode="lines",
        line=dict(color="#7c83fd", width=2, dash="dot"),
        name="Focus Score %"
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Focus Level",
        yaxis=dict(range=[0, 110], tickvals=[0, 20, 60, 100],
                   ticktext=["Drowsy", "Distracted", "At Risk", "Focused"]),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e1e2e", bordercolor="gray"),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data table (collapsible)
    with st.expander("📋 View Raw Session Data"):
        st.dataframe(df[["time", "status", "score", "noise"]].tail(20),
                     use_container_width=True)
else:
    st.info("📸 Take at least 2 photos to see your session history chart here.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("🧠 Smart Focus Analyzer | Built with OpenCV + Streamlit | "
           "For internship / portfolio use")
