# 🎬 ClipNova

Turn any YouTube video into viral short-form vertical clips — fully locally (except the Gemini API call).

## Features

- **One-click download** — pulls the best MP4 + audio from a YouTube URL
- **Local transcription** — word-level timestamps via faster-whisper (runs on your machine)
- **AI viral-moment detection** — Google Gemini picks the most engaging, self-contained moments
- **Speaker tracking** — OpenCV face detection to center the 9:16 crop on the speaker
- **Auto subtitles** — bold high-contrast subtitles burned into each clip
- **Configurable clip count & duration** — clip length defaults to exactly 10s
- **Streamlit dashboard** — preview and download each generated MP4

## Requirements

- macOS (tested) with Homebrew
- [ffmpeg-full](https://formulae.brew.sh/formula/ffmpeg) (`brew install ffmpeg-full`)
- Python 3.12+ (installed automatically by the launcher if missing)
- A free [Gemini API key](https://aistudio.google.com/apikey)

## Quick Start

```bash
./run.command
```

The launcher will:
1. Install ffmpeg + Python 3.12 if needed
2. Create the virtual environment and install dependencies
3. Open the dashboard at `http://localhost:8501`

Then paste a YouTube URL, enter your Gemini API key in the sidebar, and hit **Generate Shorts**.

## Manual setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install streamlit faster-whisper yt-dlp google-genai opencv-python-headless

# ffmpeg with libass (required for subtitle burn-in)
brew install ffmpeg-full

streamlit run app.py
```

## Usage

1. Get a free Gemini API key from https://aistudio.google.com/apikey
2. Paste a YouTube URL into the app
3. Adjust settings in the sidebar (number of clips, min/max duration)
4. Click **Generate Shorts** — clips render into `output/`

## How it works

```
YouTube URL
   │  yt-dlp (retries across player clients if blocked)
   ▼
source.mp4 + source_audio.wav
   │  faster-whisper (word-level timestamps)
   ▼
word transcript
   │  Gemini (gemini-3.6-flash, retries on 503/429)
   ▼
top viral moments (start/end/title)
   │  OpenCV speaker tracking → 9:16 crop
   ▼
FFmpeg: crop + subtitle burn-in
   │
   ▼
output/N_title.mp4
```

## Pipeline modules (`pipeline.py`)

| Function | Purpose |
|---|---|
| `download_video` | Download + extract audio, retries across yt-dlp player clients |
| `transcribe` | Word-level transcription with faster-whisper |
| `detect_viral_moments` | Gemini picks engaging clips, clamps to duration |
| `detect_speaker_center` | Face detection to find speaker X-position |
| `build_crop_filter` | Builds the 9:16 FFmpeg crop filter |
| `words_to_ass` / `words_to_srt` | Subtitle generation |
| `render_clip` | Crop, subtitle, and export each clip |
| `run_pipeline` | Full orchestration and progress callbacks |

## Notes

- **100% local processing** except the Gemini API call — only transcripts leave your machine.
- If YouTube blocks a download ("page needs to be reloaded"), the app retries automatically with alternative player clients. Persistent blocks are usually temporary rate-limiting — wait a few minutes.
- Gemini 503/429 overload errors are retried automatically with backoff.

## License

MIT