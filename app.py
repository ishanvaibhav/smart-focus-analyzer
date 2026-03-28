"""
app.py — Smart Focus Analyzer (DYNAMIC VERSION)
================================================
• Live webcam stream via streamlit-webrtc
• Auto-updates every frame — no manual clicking
• Animated status badge + gauge
• Session history chart updates in real-time
"""

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
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
# CSS — animated badges + dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@keyframes pulse-green {
  0%   { box-shadow: 0 0 0 0 rgba(57,255,20,0.5); }
  70%  { box-shadow: 0 0 0 12px rgba(57,255,20,0); }
  100% { box-shadow: 0 0 0 0 rgba(57,255,20,0); }
}
@keyframes pulse-red {
  0%   { box-shadow: 0 0 0 0 rgba(255,68,68,0.5); }
  70%  { box-shadow: 0 0 0 12px rgba(255,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,68,68,0); }
}
@keyframes pulse-orange {
  0%   { box-shadow: 0 0 0 0 rgba(255,170,0,0.5); }
  70%  { box-shadow: 0 0 0 12px rgba(255,170,0,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,170,0,0); }
}
.status-focused {
    padding: 16px; border-radius: 12px; text-align: center;
    font-size: 24px; font-weight: 900;
    background: #1a3a1a; color: #39ff14;
    border: 2px solid #39ff14;
    animation: pulse-green 1.5s infinite;
}
.status-distracted {
    padding: 16px; border-radius: 12px; text-align: center;
    font-size: 24px; font-weight: 900;
    background: #3a1a1a; color: #ff4444;
    border: 2px solid #ff4444;
    animation: pulse-red 1s infinite;
}
.status-drowsy {
    padding: 16px; border-radius: 12px; text-align: center;
    font-size: 24px; font-weight: 900;
    background: #3a2a0a; color: #ffaa00;
    border: 2px solid #ffaa00;
    animation: pulse-orange 1.2s infinite;
}
.status-at_risk {
    padding: 16px; border-radius: 12px; text-align: center;
    font-size: 24px; font-weight: 900;
    background: #2a1a0a; color: #ff8800;
    border: 2px solid #ff8800;
    animation: pulse-orange 1.2s infinite;
}
.metric-card {
    background: #1e1e2e; border-radius: 10px;
    padding: 14px 18px; margin: 6px 0;
    border-left: 4px solid #7c83fd;
}
.tip-box {
    background: #1a1a2e; border-left: 4px solid #7c83fd;
    padding: 10px 14px; border-radius: 8px; margin-top: 8px;
    font-size: 14px;
}
.stApp { background-color: #0f1117; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "history"       : [],
    "session_start" : time.time(),
    "total_frames"  : 0,
    "focused_count" : 0,
    "last_status"   : "waiting",
    "last_reason"   : "Waiting for video...",
    "last_update"   : time.time(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# VIDEO PROCESSOR — runs on every webcam frame
# ─────────────────────────────────────────────
class FocusVideoProcessor(VideoProcessorBase):
    """
    Processes each webcam frame in real-time.
    Annotates the frame and stores the result in
    st.session_state so the dashboard can read it.
    """
    def __init__(self):
        self.detector = FaceAttentionDetector()
        self.frame_count = 0
        self.SAMPLE_EVERY = 5   # Analyse every 5th frame (saves CPU)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.SAMPLE_EVERY == 0:
            result = self.detector.analyze(img, draw=True, show_eyes=True)
            status = result["status"]
            reason = result["reason"]
            img    = result["frame"]

            # Write results to session state
            st.session_state.last_status  = status
            st.session_state.last_reason  = reason
            st.session_state.last_update  = time.time()
            st.session_state.total_frames += 1

            if status == "focused":
                st.session_state.focused_count += 1

            # Append to history (max 200 points)
            score = round(
                st.session_state.focused_count /
                st.session_state.total_frames * 100, 1
            ) if st.session_state.total_frames > 0 else 0.0

            st.session_state.history.append({
                "time"  : time.strftime("%H:%M:%S"),
                "status": status,
                "score" : score,
            })
            if len(st.session_state.history) > 200:
                st.session_state.history = st.session_state.history[-200:]

        # Draw live clock + score on frame
        score = round(
            st.session_state.focused_count /
            max(st.session_state.total_frames, 1) * 100, 1
        )
        elapsed = int(time.time() - st.session_state.session_start)
        m, s = divmod(elapsed, 60)

        # HUD overlay
        h_img, w_img = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h_img - 45), (w_img, h_img), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        STATUS_COLORS = {
            "focused"   : (0, 220, 0),
            "distracted": (0, 60, 255),
            "drowsy"    : (0, 165, 255),
            "at_risk"   : (0, 140, 255),
        }
        color = STATUS_COLORS.get(st.session_state.last_status, (180, 180, 180))

        cv2.putText(img,
            f"{st.session_state.last_status.upper()}  |  Score: {score}%  |  {m:02d}:{s:02d}",
            (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=70)
    st.title("🧠 Focus Analyzer")
    st.markdown("---")

    st.subheader("⚙️ Settings")
    refresh_rate = st.slider("Dashboard Refresh (seconds)", 1, 10, 3)
    noise_val    = st.slider("🎤 Noise Level", 0, 100, 20,
                             help="Adjust to match your environment")

    if noise_val < 30:   noise_label, noise_color = "🟢 Quiet",    "#39ff14"
    elif noise_val < 65: noise_label, noise_color = "🟡 Moderate", "#ffaa00"
    else:                noise_label, noise_color = "🔴 Loud",     "#ff4444"

    st.markdown(
        f"<div style='font-size:16px;color:{noise_color};font-weight:bold'>"
        f"{noise_label}</div>", unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("📊 Live Stats")
    elapsed = int(time.time() - st.session_state.session_start)
    m, s = divmod(elapsed, 60)
    st.metric("⏱ Session", f"{m:02d}:{s:02d}")
    st.metric("📸 Frames",  st.session_state.total_frames)
    total = max(st.session_state.total_frames, 1)
    st.metric("🎯 Focus Score",
              f"{st.session_state.focused_count / total * 100:.1f}%")

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v if not callable(v) else v()
        st.session_state.session_start = time.time()
        st.success("Session reset!")

    st.markdown("---")
    st.caption("Live  •  OpenCV  •  Streamlit")

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.title("🧠 Smart Focus & Distraction Analyzer")
st.caption("Live webcam analysis — no clicking needed. Dashboard refreshes automatically.")
st.markdown("---")

col_video, col_dash = st.columns([1.3, 1], gap="large")

# ── LEFT: Live Video ───────────────────────────────────────────
with col_video:
    st.subheader("📹 Live Camera Stream")
    st.caption("Allow camera access when your browser asks.")

    webrtc_streamer(
        key="focus-stream",
        video_processor_factory=FocusVideoProcessor,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ── RIGHT: Live Dashboard ──────────────────────────────────────
with col_dash:
    st.subheader("📊 Live Dashboard")

    # Placeholders — we update these in the refresh loop
    status_placeholder = st.empty()
    reason_placeholder = st.empty()
    tip_placeholder    = st.empty()
    gauge_placeholder  = st.empty()

# ── BOTTOM: History Chart ──────────────────────────────────────
st.markdown("---")
st.subheader("📈 Real-Time Focus History")
chart_placeholder = st.empty()

# ─────────────────────────────────────────────
# LIVE REFRESH LOOP
# ─────────────────────────────────────────────
STATUS_CONFIG = {
    "focused"   : ("✅ FOCUSED",    "You're in the zone! Keep it up."),
    "distracted": ("❌ DISTRACTED", "Look directly at the screen."),
    "drowsy"    : ("😴 DROWSY",     "Take a short break — splash water on your face."),
    "at_risk"   : ("⚠️ AT RISK",    "Noisy environment — try moving somewhere quieter."),
    "waiting"   : ("⏳ WAITING",    "Start the camera stream above."),
}

STATUS_SCORE_MAP = {"focused": 100, "at_risk": 60, "distracted": 20, "drowsy": 10}
STATUS_COLORS_MAP = {
    "focused"   : "#39ff14",
    "distracted": "#ff4444",
    "drowsy"    : "#ffaa00",
    "at_risk"   : "#ff8800",
}

while True:
    status = st.session_state.last_status
    reason = st.session_state.last_reason

    # Apply noise override
    if noise_val >= 65 and status == "focused":
        status = "at_risk"
        reason = "Focused but loud environment"

    label, tip = STATUS_CONFIG.get(status, ("❓ UNKNOWN", "Try again."))

    # ── Animated Status Badge ──────────────────────────────────
    with status_placeholder:
        st.markdown(
            f'<div class="status-{status}">{label}</div>',
            unsafe_allow_html=True
        )

    with reason_placeholder:
        st.markdown(f"<small style='color:#aaa'>📍 {reason}</small>",
                    unsafe_allow_html=True)

    with tip_placeholder:
        st.markdown(
            f'<div class="tip-box">💡 {tip}</div>',
            unsafe_allow_html=True
        )

    # ── Animated Gauge ────────────────────────────────────────
    total = max(st.session_state.total_frames, 1)
    score = st.session_state.focused_count / total * 100

    gauge_color = (
        "#39ff14" if score >= 70 else
        "#ffaa00" if score >= 40 else
        "#ff4444"
    )

    gauge = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = score,
        delta  = {"reference": 70, "valueformat": ".1f"},
        title  = {"text": "Focus Score %", "font": {"size": 14, "color": "white"}},
        number = {"suffix": "%", "font": {"size": 44, "color": "white"}},
        gauge  = {
            "axis"      : {"range": [0, 100], "tickcolor": "white"},
            "bar"       : {"color": gauge_color},
            "bgcolor"   : "#1e1e2e",
            "bordercolor": "#444",
            "steps": [
                {"range": [0,  40], "color": "#2a0a0a"},
                {"range": [40, 70], "color": "#2a1a00"},
                {"range": [70,100], "color": "#0a2a0a"},
            ],
            "threshold": {
                "line" : {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": 70
            }
        }
    ))
    gauge.update_layout(
        height=230,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0f1117",
        font=dict(color="white")
    )
    gauge_placeholder.plotly_chart(gauge, use_container_width=True)

    # ── Auto-updating History Chart ────────────────────────────
    if len(st.session_state.history) >= 2:
        df = pd.DataFrame(st.session_state.history)
        df["value"] = df["status"].map(STATUS_SCORE_MAP).fillna(50)

        fig = go.Figure()

        for s, color in STATUS_COLORS_MAP.items():
            mask = df["status"] == s
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df[mask]["time"],
                    y=df[mask]["value"],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="circle"),
                    name=s.capitalize()
                ))

        # Rolling focus score line
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["score"],
            mode="lines",
            line=dict(color="#7c83fd", width=2, dash="dot"),
            name="Focus %"
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(
                range=[0, 110],
                tickvals=[10, 20, 60, 100],
                ticktext=["Drowsy", "Distracted", "At Risk", "Focused"],
                tickcolor="white", color="white"
            ),
            xaxis=dict(tickcolor="white", color="white"),
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="white"),
            legend=dict(bgcolor="#1e1e2e", bordercolor="#444"),
            margin=dict(t=10, b=40),
            height=300,
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        chart_placeholder.info(
            "📹 Start the camera stream — history chart will appear here automatically."
        )

    # ── Wait then auto-refresh ─────────────────────────────────
    time.sleep(refresh_rate)
    st.rerun()
