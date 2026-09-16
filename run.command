#!/bin/bash
# ClipNova - One-click launcher
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Ensure ffmpeg exists (needed for ASS subtitles via ffmpeg-full)
if ! [ -x /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg ] && ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Installing ffmpeg... (brew install ffmpeg-full)"
    brew install ffmpeg-full
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    PY=/opt/homebrew/bin/python3.12
    if [ ! -x "$PY" ]; then PY=python3; fi
    "$PY" -m venv venv
fi

source venv/bin/activate
pip install -q --upgrade pip

# Install deps only when missing
python -c "import streamlit" 2>/dev/null || pip install -q streamlit
python -c "import yt_dlp" 2>/dev/null || pip install -q yt-dlp
python -c "import faster_whisper" 2>/dev/null || pip install -q faster-whisper
python -c "import genai" 2>/dev/null || pip install -q google-genai
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless
python -c "import mediapipe" 2>/dev/null || pip install -q mediapipe

echo ""
echo "🎬 Starting ClipNova..."
echo "   Dashboard will open at http://localhost:8501"
echo "   Press Ctrl+C to stop."
echo ""

# Open browser after short delay
(sleep 3 && open "http://localhost:8501") &

streamlit run app.py --server.headless false