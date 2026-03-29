"""
Smart Focus Analyzer.

This version keeps the WebRTC stream stable and refreshes the dashboard
without writing to Streamlit session state from the video processing thread.
"""

import threading
import time

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detector import FaceAttentionDetector

st.set_page_config(
    page_title="Smart Focus Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
except ModuleNotFoundError as exc:
    st.error("Live streaming dependencies are missing in this Python environment.")
    st.code("pip install -r requirements.txt")
    st.caption(f"Missing module: {exc.name}")
    st.stop()


st.markdown(
    """
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
    .tip-box {
        background: #1a1a2e; border-left: 4px solid #7c83fd;
        padding: 10px 14px; border-radius: 8px; margin-top: 8px;
        font-size: 14px;
    }
    .stApp { background-color: #0f1117; }
    </style>
    """,
    unsafe_allow_html=True,
)


def default_metrics(session_start=None):
    return {
        "history": [],
        "session_start": session_start or time.time(),
        "total_frames": 0,
        "focused_count": 0,
        "last_status": "waiting",
        "last_reason": "Start the camera stream above.",
        "last_update": None,
    }


class FocusVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = FaceAttentionDetector()
        self.sample_every = 5
        self._frame_count = 0
        self._lock = threading.Lock()
        self._metrics = default_metrics(session_start=time.time())

    def reset(self):
        with self._lock:
            self._frame_count = 0
            self._metrics = default_metrics(session_start=time.time())

    def get_snapshot(self):
        with self._lock:
            snapshot = dict(self._metrics)
            snapshot["history"] = [entry.copy() for entry in self._metrics["history"]]
        return snapshot

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        if self._frame_count % self.sample_every == 0:
            result = self.detector.analyze(img, draw=True, show_eyes=True)
            img = result["frame"]
            self._record_result(result["status"], result["reason"])

        score, session_start, status = self._overlay_state()
        elapsed = int(time.time() - session_start)
        minutes, seconds = divmod(elapsed, 60)

        overlay = img.copy()
        height, width = img.shape[:2]
        cv2.rectangle(overlay, (0, height - 45), (width, height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        status_colors = {
            "focused": (0, 220, 0),
            "distracted": (0, 60, 255),
            "drowsy": (0, 165, 255),
            "at_risk": (0, 140, 255),
        }
        color = status_colors.get(status, (180, 180, 180))

        cv2.putText(
            img,
            f"{status.upper()} | Score: {score}% | {minutes:02d}:{seconds:02d}",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _record_result(self, status, reason):
        with self._lock:
            metrics = self._metrics
            metrics["last_status"] = status
            metrics["last_reason"] = reason
            metrics["last_update"] = time.time()
            metrics["total_frames"] += 1

            if status == "focused":
                metrics["focused_count"] += 1

            total_frames = metrics["total_frames"]
            score = round(metrics["focused_count"] / total_frames * 100, 1) if total_frames else 0.0

            metrics["history"].append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "status": status,
                    "score": score,
                }
            )
            if len(metrics["history"]) > 200:
                metrics["history"] = metrics["history"][-200:]

    def _overlay_state(self):
        with self._lock:
            focused_count = self._metrics["focused_count"]
            total_frames = self._metrics["total_frames"]
            session_start = self._metrics["session_start"]
            status = self._metrics["last_status"]

        score = round(focused_count / max(total_frames, 1) * 100, 1)
        return score, session_start, status


STATUS_CONFIG = {
    "focused": ("FOCUSED", "You're in the zone. Keep going."),
    "distracted": ("DISTRACTED", "Look directly at the screen."),
    "drowsy": ("DROWSY", "Take a short break and reset."),
    "at_risk": ("AT RISK", "The environment is noisy. Try a quieter spot."),
    "waiting": ("WAITING", "Start the camera stream above."),
}

STATUS_SCORE_MAP = {
    "focused": 100,
    "at_risk": 60,
    "distracted": 20,
    "drowsy": 10,
}

STATUS_COLORS_MAP = {
    "focused": "#39ff14",
    "distracted": "#ff4444",
    "drowsy": "#ffaa00",
    "at_risk": "#ff8800",
}


def get_snapshot(context):
    if context and context.state.playing and context.video_processor:
        return context.video_processor.get_snapshot()
    return default_metrics()


def display_state(snapshot, noise_level):
    status = snapshot["last_status"]
    reason = snapshot["last_reason"]

    if noise_level >= 65 and status == "focused":
        status = "at_risk"
        reason = "Focused but the environment is noisy."

    label, tip = STATUS_CONFIG.get(status, ("UNKNOWN", "Try again."))
    return status, reason, label, tip


def focus_score(snapshot):
    total_frames = snapshot["total_frames"]
    if total_frames == 0:
        return 0.0
    return round(snapshot["focused_count"] / total_frames * 100, 1)


def build_rtc_configuration():
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    secret_ice_servers = st.secrets.get("ice_servers")
    if secret_ice_servers:
        ice_servers = list(secret_ice_servers)
    else:
        turn_url = st.secrets.get("turn_server_url")
        turn_username = st.secrets.get("turn_username")
        turn_password = st.secrets.get("turn_password")
        if turn_url and turn_username and turn_password:
            ice_servers.append(
                {
                    "urls": [turn_url],
                    "username": turn_username,
                    "credential": turn_password,
                }
            )

    return RTCConfiguration({"iceServers": ice_servers})


def build_gauge(score):
    gauge_color = "#39ff14" if score >= 70 else "#ffaa00" if score >= 40 else "#ff4444"

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 70, "valueformat": ".1f"},
            title={"text": "Focus Score %", "font": {"size": 14, "color": "white"}},
            number={"suffix": "%", "font": {"size": 44, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": gauge_color},
                "bgcolor": "#1e1e2e",
                "bordercolor": "#444",
                "steps": [
                    {"range": [0, 40], "color": "#2a0a0a"},
                    {"range": [40, 70], "color": "#2a1a00"},
                    {"range": [70, 100], "color": "#0a2a0a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    figure.update_layout(
        height=230,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0f1117",
        font=dict(color="white"),
    )
    return figure


def build_history_chart(history):
    df = pd.DataFrame(history)
    df["value"] = df["status"].map(STATUS_SCORE_MAP).fillna(50)

    figure = go.Figure()

    for status, color in STATUS_COLORS_MAP.items():
        mask = df["status"] == status
        if mask.any():
            figure.add_trace(
                go.Scatter(
                    x=df.loc[mask, "time"],
                    y=df.loc[mask, "value"],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="circle"),
                    name=status.capitalize(),
                )
            )

    figure.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["score"],
            mode="lines",
            line=dict(color="#7c83fd", width=2, dash="dot"),
            name="Focus %",
        )
    )

    figure.update_layout(
        xaxis_title="Time",
        yaxis=dict(
            range=[0, 110],
            tickvals=[10, 20, 60, 100],
            ticktext=["Drowsy", "Distracted", "At Risk", "Focused"],
            tickcolor="white",
            color="white",
        ),
        xaxis=dict(tickcolor="white", color="white"),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e1e2e", bordercolor="#444"),
        margin=dict(t=10, b=40),
        height=300,
    )
    return figure


def render_live_sections(
    snapshot,
    noise_level,
    session_metric,
    frames_metric,
    focus_metric_placeholder,
    status_placeholder,
    reason_placeholder,
    tip_placeholder,
    gauge_placeholder,
    chart_placeholder,
):
    status, reason, label, tip = display_state(snapshot, noise_level)
    score = focus_score(snapshot)

    elapsed = 0
    if snapshot["total_frames"] > 0 or status != "waiting":
        elapsed = int(time.time() - snapshot["session_start"])
    minutes, seconds = divmod(elapsed, 60)

    session_metric.metric("Session", f"{minutes:02d}:{seconds:02d}")
    frames_metric.metric("Frames", snapshot["total_frames"])
    focus_metric_placeholder.metric("Focus Score", f"{score:.1f}%")

    status_placeholder.markdown(
        f'<div class="status-{status}">{label}</div>',
        unsafe_allow_html=True,
    )
    reason_placeholder.markdown(
        f"<small style='color:#aaa'>{reason}</small>",
        unsafe_allow_html=True,
    )
    tip_placeholder.markdown(
        f'<div class="tip-box">{tip}</div>',
        unsafe_allow_html=True,
    )
    gauge_placeholder.plotly_chart(build_gauge(score), use_container_width=True)

    if len(snapshot["history"]) >= 2:
        chart_placeholder.plotly_chart(
            build_history_chart(snapshot["history"]),
            use_container_width=True,
        )
    else:
        chart_placeholder.info(
            "Start the camera stream to build the live focus history chart."
        )


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=70)
    st.title("Focus Analyzer")
    st.markdown("---")

    st.subheader("Settings")
    refresh_rate = st.slider("Dashboard Refresh (seconds)", 1, 10, 2)
    noise_level = st.slider(
        "Noise Level",
        0,
        100,
        20,
        help="Use this to match the environment around you.",
    )

    if noise_level < 30:
        noise_label, noise_color = "Quiet", "#39ff14"
    elif noise_level < 65:
        noise_label, noise_color = "Moderate", "#ffaa00"
    else:
        noise_label, noise_color = "Loud", "#ff4444"

    st.markdown(
        (
            "<div style='font-size:16px;color:"
            f"{noise_color};font-weight:bold'>{noise_label}</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Live Stats")
    session_metric = st.empty()
    frames_metric = st.empty()
    focus_metric_placeholder = st.empty()

    st.markdown("---")
    reset_clicked = st.button("Reset Session", use_container_width=True)
    reset_message = st.empty()

    st.markdown("---")
    st.caption("Live | OpenCV | Streamlit")


st.title("Smart Focus & Distraction Analyzer")
st.caption("Stable live webcam analysis with an auto-refreshing dashboard.")
st.markdown("---")

col_video, col_dash = st.columns([1.3, 1], gap="large")

with col_video:
    st.subheader("Live Camera Stream")
    st.caption("Allow camera access when your browser asks.")
    st.caption(
        "If you deploy this app remotely, add TURN credentials in Streamlit secrets "
        "for the most reliable connection."
    )

    context = webrtc_streamer(
        key="focus-stream",
        video_processor_factory=FocusVideoProcessor,
        rtc_configuration=build_rtc_configuration(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if not context.state.playing:
        st.info("Click Start to begin the live stream.")

with col_dash:
    st.subheader("Live Dashboard")
    status_placeholder = st.empty()
    reason_placeholder = st.empty()
    tip_placeholder = st.empty()
    gauge_placeholder = st.empty()

st.markdown("---")
st.subheader("Real-Time Focus History")
chart_placeholder = st.empty()

if reset_clicked:
    if context and context.video_processor:
        context.video_processor.reset()
        reset_message.success("Session reset.")
    else:
        reset_message.info("Start the camera stream first.")

render_live_sections(
    get_snapshot(context),
    noise_level,
    session_metric,
    frames_metric,
    focus_metric_placeholder,
    status_placeholder,
    reason_placeholder,
    tip_placeholder,
    gauge_placeholder,
    chart_placeholder,
)

while context.state.playing:
    render_live_sections(
        get_snapshot(context),
        noise_level,
        session_metric,
        frames_metric,
        focus_metric_placeholder,
        status_placeholder,
        reason_placeholder,
        tip_placeholder,
        gauge_placeholder,
        chart_placeholder,
    )
    time.sleep(refresh_rate)
