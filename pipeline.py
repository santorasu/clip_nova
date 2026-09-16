from __future__ import annotations

import os
import json
import subprocess
import tempfile
import re
import time
from pathlib import Path
from typing import Optional

import shutil

import yt_dlp
from faster_whisper import WhisperModel
from google import genai
import cv2
import numpy as np


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary — prefer ffmpeg-full (has libass) if available."""
    for candidate in [
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
        shutil.which("ffmpeg"),
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise RuntimeError("ffmpeg not found. Install via: brew install ffmpeg-full")


FFMPEG = _find_ffmpeg()


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: str) -> dict:
    """Download best MP4 + extract audio. Returns dict with paths."""
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "source.mp4")
    audio_path = os.path.join(output_dir, "source_audio.wav")

    clients = ["android", "tv", "web", "ios", "mweb", "tv_embedded"]

    last_err: Optional[Exception] = None
    info: Optional[dict] = None
    for attempt in range(2):
        for client in clients:
            ydl_opts = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl": os.path.join(output_dir, "source.%(ext)s"),
                "merge_output_format": "mp4",
                "extractor_args": {"youtube": [f"player_client={client}"]},
                "quiet": True,
                "no_warnings": True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    break
            except Exception as e:  # noqa: BLE001 - try next client on failure
                last_err = e
                time.sleep(5)
        if info is not None:
            break
    if info is None:
        raise RuntimeError(
            "YouTube is blocking downloads (page needs to be reloaded). "
            "Wait a few minutes and retry. "
            f"Last error: {last_err}"
        )
    title = info.get("title", "video")

    if not os.path.exists(video_path):
        for f in os.listdir(output_dir):
            if f.startswith("source.") and f.endswith((".mp4", ".mkv", ".webm")):
                candidate = os.path.join(output_dir, f)
                if candidate != video_path:
                    os.rename(candidate, video_path)
                break

    subprocess.run(
        [
            FFMPEG, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path,
        ],
        check=True,
        capture_output=True,
    )
    return {"video": video_path, "audio": audio_path, "title": title}


# ---------------------------------------------------------------------------
# 2. Transcription
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def transcribe(audio_path: str, model_size: str = "base") -> list[dict]:
    """Transcribe audio with faster-whisper. Returns list of word dicts."""
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
    model = _model_cache[model_size]

    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=5,
        language="en",
    )

    words: list[dict] = []
    for seg in segments:
        for w in seg.words:
            words.append({
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
            })
    return words


# ---------------------------------------------------------------------------
# 3. AI Viral Moment Detection via Gemini
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are a viral video clip editor. Given a word-level timestamped transcript, \
identify the {n} most engaging, self-contained moments suitable for short-form vertical video.

Return ONLY a valid JSON array (no markdown fences, no commentary). Each element:
{{
  "start": <float seconds>,
  "end": <float seconds>,
  "title": "<catchy 3-8 word title>"
}}

Rules:
- Each clip should be {min_dur}-{max_dur} seconds.
- Moments must be self-contained (make sense without context).
- Prioritize humor, insight, emotional peaks, strong statements.
- Start/end times must align with word boundaries below.

Transcript (word -> timestamp):
"""


def detect_viral_moments(
    words: list[dict],
    api_key: str,
    n: int = 3,
    min_dur: float = 10,
    max_dur: float = 10,
) -> list[dict]:
    """Use Gemini Flash to pick the top N viral moments."""
    transcript_lines = []
    for w in words:
        transcript_lines.append(f"[{w['start']:.2f}-{w['end']:.2f}] {w['word']}")
    transcript_text = "\n".join(transcript_lines)

    prompt = (
        _PROMPT_TEMPLATE.format(n=n, min_dur=min_dur, max_dur=max_dur)
        + transcript_text
        + f"\n\nConstraints: each clip exactly {min_dur} seconds, "
          f"return exactly {n} clips."
    )

    client = genai.Client(api_key=api_key)
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            break
        except Exception as e:  # noqa: BLE001 - transient API errors
            last_err = e
            is_transient = "503" in str(e) or "429" in str(e) or "500" in str(e)
            if not is_transient:
                raise
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(
            f"Gemini API busy after 5 attempts. Wait a few minutes and retry. {last_err}"
        )

    raw = response.text.strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    clips = json.loads(raw)

    # Clamp durations
    for c in clips:
        c["start"] = max(0, float(c["start"]))
        c["end"] = max(c["start"] + min_dur, float(c["end"]))
        if c["end"] - c["start"] > max_dur:
            c["end"] = c["start"] + max_dur
    return clips


# ---------------------------------------------------------------------------
# 4. Speaker Tracking & 9:16 Reframe
# ---------------------------------------------------------------------------

def detect_speaker_center(video_path: str, sample_frames: int = 30) -> float:
    """Sample frames, run face detection, return average X-center (0-1 normalized)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0.5

    indices = np.linspace(0, max(total - 1, 0), sample_frames, dtype=int)
    centers: list[float] = []

    cascade_path = os.path.join(os.path.dirname(__file__), "cascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = largest
            cx = (fx + fw / 2) / frame.shape[1]
            centers.append(cx)
    cap.release()

    if not centers:
        return 0.5
    return float(np.median(centers))


def build_crop_filter(video_path: str, speaker_x: float | None = None) -> str:
    """Return FFmpeg -vf crop filter string for 9:16 centered on speaker."""
    if speaker_x is None:
        speaker_x = detect_speaker_center(video_path)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    target_w = int(h * 9 / 16)
    if target_w > w:
        target_w = w

    center_px = int(speaker_x * w)
    x = max(0, min(center_px - target_w // 2, w - target_w))

    return f"crop={target_w}:{h}:{x}:0"


# ---------------------------------------------------------------------------
# 5. SRT / ASS Subtitle Generation & Burn-in
# ---------------------------------------------------------------------------

def words_to_srt(words: list[dict], start: float, end: float) -> str:
    """Create SRT content for words within [start, end] window."""
    relevant = [w for w in words if w["end"] > start and w["start"] < end]
    if not relevant:
        return ""

    # Group into ~6 word chunks for readability
    lines: list[str] = []
    idx = 1
    chunk: list[dict] = []
    for w in relevant:
        chunk.append(w)
        if len(chunk) >= 6 or (chunk and w["word"].endswith((".", "!", "?", ","))):
            lines.append(chunk)
            chunk = []
    if chunk:
        lines.append(chunk)

    srt_parts: list[str] = []
    for i, group in enumerate(lines):
        s_start = max(0, group[0]["start"] - start)
        s_end = group[-1]["end"] - start
        text = " ".join(w["word"] for w in group)
        srt_parts.append(
            f"{i + 1}\n"
            f"{_srt_ts(s_start)} --> {_srt_ts(s_end)}\n"
            f"{text}\n"
        )
    return "\n".join(srt_parts)


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_ass(words: list[dict], start: float, end: float, width: int, height: int) -> str:
    """Create ASS subtitle content with bold high-contrast styling."""
    relevant = [w for w in words if w["end"] > start and w["start"] < end]
    if not relevant:
        return ""

    # Group into ~5 word chunks
    lines: list[str] = []
    chunk: list[dict] = []
    for w in relevant:
        chunk.append(w)
        if len(chunk) >= 5 or (chunk and w["word"].endswith((".", "!", "?", ","))):
            lines.append(chunk)
            chunk = []
    if chunk:
        lines.append(chunk)

    ass_events: list[str] = []
    for group in lines:
        s_start = max(0, group[0]["start"] - start)
        s_end = group[-1]["end"] - start
        text = " ".join(w["word"] for w in group)
        ass_events.append(
            f"Dialogue: 0,{_ass_ts(s_start)},{_ass_ts(s_end)},"
            f"Default,,0,0,0,,{text}"
        )

    header = (
        "[Script Info]\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "ScriptType: v4.00+\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
        "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial Black,28,&H00FFFFFF,&H000000FF,"
        "&H00000000,&H80000000,-1,0,0,0,"
        "100,100,0,0,1,3,1,"
        "2,20,20,50,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    return header + "\n".join(ass_events)


def _ass_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# 6. Render clip with FFmpeg
# ---------------------------------------------------------------------------

def render_clip(
    video_path: str,
    words: list[dict],
    start: float,
    end: float,
    crop_filter: str,
    output_path: str,
    title: str = "",
) -> str:
    """Crop, add subtitles, burn-in, and export a single clip."""
    # Write ASS subtitle file
    cap = cv2.VideoCapture(video_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ass_content = words_to_ass(words, start, end, vw, vh)
    ass_path = output_path + ".ass"
    with open(ass_path, "w") as f:
        f.write(ass_content)

    duration = end - start

    cmd = [
        FFMPEG, "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"{crop_filter},ass={ass_path}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    url: str,
    api_key: str,
    output_dir: str,
    n_clips: int = 3,
    min_dur: float = 10,
    max_dur: float = 10,
    progress_callback=None,
) -> list[dict]:
    """Execute the full pipeline and return list of clip info dicts."""
    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    temp_dir = tempfile.mkdtemp(prefix="clipnova_")

    # Step 1 - Download
    emit("Downloading video...")
    dl = download_video(url, temp_dir)
    emit(f"Downloaded: {dl['title']}")

    # Step 2 - Transcribe
    emit("Transcribing audio (faster-whisper)...")
    words = transcribe(dl["audio"], model_size="base")
    emit(f"Transcribed {len(words)} words")

    # Step 3 - AI analysis
    emit("AI analysis (Gemini Flash)...")
    clips = detect_viral_moments(
        words, api_key, n=n_clips, min_dur=min_dur, max_dur=max_dur
    )
    emit(f"Found {len(clips)} viral moments")

    # Step 4 - Speaker tracking
    emit("Detecting speaker position...")
    speaker_x = detect_speaker_center(dl["video"])
    crop_filter = build_crop_filter(dl["video"], speaker_x)
    emit(f"Speaker center: {speaker_x:.2f}, crop: {crop_filter}")

    # Step 5 - Render each clip
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []

    for i, clip in enumerate(clips):
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", clip.get("title", f"clip_{i}"))[:40]
        out_file = os.path.join(output_dir, f"{i+1}_{safe_title}.mp4")
        emit(f"Rendering clip {i+1}/{len(clips)}: {clip['title']}...")

        render_clip(
            video_path=dl["video"],
            words=words,
            start=clip["start"],
            end=clip["end"],
            crop_filter=crop_filter,
            output_path=out_file,
            title=clip["title"],
        )
        results.append({
            "path": out_file,
            "title": clip["title"],
            "start": clip["start"],
            "end": clip["end"],
        })
        emit(f"Done: {out_file}")

    emit("All clips rendered!")
    return results
