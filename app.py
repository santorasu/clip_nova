import os
import sys
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClipNova - AI Video Clipper",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: #0e0e10; }
    .clip-card {
        background: #1a1a2e; border-radius: 12px; padding: 16px;
        margin-bottom: 16px; border: 1px solid #2a2a4a;
    }
    .clip-title {
        color: #e0e0ff; font-size: 1.1em; font-weight: bold; margin: 8px 0;
    }
    .time-badge {
        background: #3a3a6a; color: #b0b0ff; padding: 2px 10px;
        border-radius: 12px; font-size: 0.8em; display: inline-block;
    }
    div[data-testid="stMetricValue"] { color: #7c7cff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your free key from https://aistudio.google.com/apikey",
    )
    st.markdown("---")
    n_clips = st.slider("Number of clips", 1, 5, 3)
    min_dur = st.slider("Min clip duration (s)", 5, 30, 10)
    max_dur = st.slider("Max clip duration (s)", 10, 60, 10)
    st.markdown("---")
    st.caption("100% local processing. Only Gemini API calls leave your machine.")

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown("# 🎬 ClipNova")
st.markdown("Turn any YouTube video into viral short-form clips — locally.")

url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

col1, col2 = st.columns([1, 4])
with col1:
    generate = st.button("🚀 Generate Shorts", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if generate:
    if not api_key:
        st.error("Enter your Gemini API key in the sidebar.")
        st.stop()
    if not url.strip():
        st.error("Paste a YouTube URL.")
        st.stop()

    output_dir = str(Path(__file__).parent / "output")
    status_area = st.empty()
    progress_bar = st.progress(0, text="Starting...")

    stages = {
        "Downloading": 0.1,
        "Transcribing": 0.3,
        "AI analysis": 0.5,
        "Detecting speaker": 0.65,
        "Rendering clip": 0.75,
        "Done": 1.0,
    }

    def on_progress(msg: str):
        status_area.info(f"⏳ {msg}")
        for key, val in stages.items():
            if key.lower() in msg.lower():
                progress_bar.progress(val, text=msg)
                break

    try:
        results = run_pipeline(
            url=url.strip(),
            api_key=api_key,
            output_dir=output_dir,
            n_clips=n_clips,
            min_dur=float(min_dur),
            max_dur=float(max_dur),
            progress_callback=on_progress,
        )
        progress_bar.progress(1.0, text="Complete!")
        status_area.success(f"Generated {len(results)} clip(s)!")

        # Results gallery
        st.markdown("---")
        st.markdown("## 📹 Generated Clips")
        cols = st.columns(min(len(results), 3))

        for i, clip in enumerate(results):
            col = cols[i % len(cols)]
            with col:
                st.markdown(
                    f'<div class="clip-card">'
                    f'<div class="clip-title">{clip["title"]}</div>'
                    f'<span class="time-badge">{clip["start"]:.1f}s — {clip["end"]:.1f}s</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if os.path.exists(clip["path"]):
                    st.video(clip["path"])
                    with open(clip["path"], "rb") as f:
                        st.download_button(
                            label="⬇️ Download MP4",
                            data=f,
                            file_name=os.path.basename(clip["path"]),
                            mime="video/mp4",
                            key=f"dl_{i}",
                        )

    except Exception as e:
        progress_bar.progress(0)
        status_area.error(f"Pipeline failed: {e}")
        st.exception(e)
