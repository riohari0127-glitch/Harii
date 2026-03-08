import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
import re
import os
import json
import tempfile
import subprocess
import shutil

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Video Summarizer", page_icon="🎓", layout="centered")

st.title("🎓 Intelligent Video Summarizer")
st.caption("Upload a video or paste a YouTube link → get a short summary video + study notes.")

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣  Extract Video ID
# ─────────────────────────────────────────────────────────────────────────────
def extract_video_id(url: str):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣  Load Whisper and transcribe WITH timestamps
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    import whisper
    return whisper.load_model("base")

def transcribe_with_timestamps(video_path: str):
    """Returns full text + list of segments with start/end times"""
    audio_path = video_path.replace(".mp4", "_audio.mp3").replace(".mkv", "_audio.mp3") \
                           .replace(".avi", "_audio.mp3").replace(".mov", "_audio.mp3") \
                           .replace(".webm", "_audio.mp3")
    audio_path = os.path.splitext(video_path)[0] + "_audio.mp3"

    # Extract audio
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-ab", "128k",
        "-y", audio_path
    ], capture_output=True, timeout=180)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        return None, None

    whisper_model = load_whisper()
    result = whisper_model.transcribe(audio_path, language="en", fp16=False, verbose=False)

    full_text = result.get("text", "").strip()
    segments  = result.get("segments", [])  # each has: start, end, text

    return full_text, segments

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  Summarize text using BART
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_summarizer():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

def chunk_text(text: str, max_words: int = 400):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def summarize_full_text(text: str):
    import torch
    tokenizer, model = load_summarizer()
    chunks = chunk_text(text)
    summaries = []
    bar = st.progress(0, text="Summarizing transcript...")
    for i, chunk in enumerate(chunks):
        if len(chunk.split()) < 30:
            bar.progress((i+1)/len(chunks))
            continue
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                max_length=250, min_length=80,
                length_penalty=2.0, num_beams=4, early_stopping=True
            )
        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        bar.progress((i+1)/len(chunks), text=f"Summarizing chunk {i+1}/{len(chunks)}...")
    bar.empty()
    return " ".join(summaries)

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣  Match summary sentences → find timestamps in segments
# ─────────────────────────────────────────────────────────────────────────────
def find_segment_times(summary_sentences: list, segments: list, context_seconds: float = 3.0):
    """
    For each summary sentence, find which segment it best matches
    and return (start_time, end_time) for that clip.
    """
    clips = []
    used_ranges = []

    for sentence in summary_sentences:
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        best_score = 0
        best_seg   = None

        for seg in segments:
            seg_words = set(re.findall(r'\w+', seg["text"].lower()))
            # Overlap score: how many words match
            overlap = len(sentence_words & seg_words)
            score   = overlap / max(len(sentence_words), 1)
            if score > best_score:
                best_score = score
                best_seg   = seg

        if best_seg and best_score > 0.15:  # at least 15% word overlap
            start = max(0, best_seg["start"] - 1.0)
            end   = best_seg["end"] + context_seconds

            # Avoid duplicate/overlapping clips
            overlap_found = False
            for (us, ue) in used_ranges:
                if start < ue and end > us:
                    overlap_found = True
                    break

            if not overlap_found:
                clips.append((start, end, sentence))
                used_ranges.append((start, end))

    # Sort clips by time
    clips.sort(key=lambda x: x[0])
    return clips

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣  Cut clips and merge into summary video
# ─────────────────────────────────────────────────────────────────────────────
def create_summary_video(video_path: str, clips: list, output_path: str) -> bool:
    """
    Cuts clips from original video and concatenates them.
    Returns True if successful.
    """
    if not clips:
        return False

    tmpdir = os.path.dirname(output_path)
    clip_paths = []

    for i, (start, end, _) in enumerate(clips):
        duration = end - start
        if duration <= 0:
            continue
        clip_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
        result = subprocess.run([
            "ffmpeg",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            "-y", clip_path
        ], capture_output=True, timeout=120)

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            clip_paths.append(clip_path)

    if not clip_paths:
        return False

    # Write concat list
    concat_file = os.path.join(tmpdir, "concat.txt")
    with open(concat_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    # Merge all clips
    result = subprocess.run([
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y", output_path
    ], capture_output=True, timeout=300)

    return os.path.exists(output_path) and os.path.getsize(output_path) > 1000

# ─────────────────────────────────────────────────────────────────────────────
# 6️⃣  Build Study Notes text
# ─────────────────────────────────────────────────────────────────────────────
def build_study_notes(summary_text: str, word_count: int, source: str, total_duration: float, clip_count: int) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text) if len(s.strip()) > 30]
    sentences = [s if s[-1] in ".!?" else s+"." for s in sentences]

    overview   = sentences[0] if sentences else "No overview available."
    key_points = sentences[1:] if len(sentences) > 1 else sentences

    md  = "---\n## 📋 Video Overview\n\n"
    md += f"> {overview}\n\n"
    md += "---\n## 📌 Key Points\n\n"
    for pt in key_points:
        md += f"- {pt}\n"
    md += "\n---\n## 📊 Summary Stats\n\n"
    md += f"| Item | Value |\n|---|---|\n"
    md += f"| Source | {source} |\n"
    md += f"| Original Length | {int(total_duration//60)}m {int(total_duration%60)}s |\n"
    md += f"| Summary Clips | {clip_count} |\n"
    md += f"| Transcript Words | {word_count} |\n"
    md += "\n---\n"
    md += "> 💡 *Tip: Watch the summary video first, then read the key points for full understanding.*\n"
    return md

# ─────────────────────────────────────────────────────────────────────────────
# 7️⃣  Get video duration
# ─────────────────────────────────────────────────────────────────────────────
def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 8️⃣  Full pipeline for uploaded video
# ─────────────────────────────────────────────────────────────────────────────
def process_uploaded_video(video_bytes: bytes, filename: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save video
        video_path = os.path.join(tmpdir, filename)
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        duration = get_video_duration(video_path)

        # Step 1: Transcribe with timestamps
        st.info("🎤 Step 1/3 — Transcribing audio with Whisper AI...")
        full_text, segments = transcribe_with_timestamps(video_path)

        if not full_text or not segments:
            st.error("Could not transcribe audio. Make sure the video has speech.")
            return

        word_count = len(full_text.split())
        st.success(f"✅ Transcription done — {word_count} words, {len(segments)} segments")

        # Step 2: Summarize
        st.info("🤖 Step 2/3 — AI is summarizing the transcript...")
        summary_text = summarize_full_text(full_text)

        # Extract key sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text) if len(s.strip()) > 30]
        sentences = [s if s[-1] in ".!?" else s+"." for s in sentences]

        # Step 3: Create summary video
        st.info("✂️ Step 3/3 — Cutting key clips and building summary video...")
        clips = find_segment_times(sentences, segments, context_seconds=4.0)

        if not clips:
            st.warning("⚠️ Could not find matching video clips. Showing study notes only.")
        else:
            output_video_path = os.path.join(tmpdir, "summary_video.mp4")
            success = create_summary_video(video_path, clips, output_video_path)

            if success:
                # Copy to a stable location for display
                stable_path = os.path.join(tempfile.gettempdir(), "summary_video_output.mp4")
                shutil.copy2(output_video_path, stable_path)

                st.success(f"🎬 Summary video created — {len(clips)} key clips extracted!")
                st.divider()
                st.subheader("🎬 Summary Video")
                st.caption(f"This video contains the {len(clips)} most important moments from your original video.")

                with open(stable_path, "rb") as vf:
                    video_data = vf.read()

                st.video(video_data)

                st.download_button(
                    label="⬇️ Download Summary Video (.mp4)",
                    data=video_data,
                    file_name="summary_video.mp4",
                    mime="video/mp4"
                )
            else:
                st.warning("⚠️ Could not create summary video. Showing study notes only.")

        # Always show study notes too
        notes = build_study_notes(summary_text, word_count, filename, duration, len(clips) if clips else 0)
        st.divider()
        st.subheader("📚 Study Notes")
        st.markdown(notes)

        st.download_button(
            label="⬇️ Download Study Notes (.txt)",
            data=notes,
            file_name="study_notes.txt",
            mime="text/plain",
            key="notes_dl"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 9️⃣  YouTube URL Tab helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_transcript_youtube(video_url: str):
    subtitle_extensions = [".json3", ".vtt", ".webvtt", ".srt"]
    browsers = ["chrome", "firefox", "edge", None]

    with tempfile.TemporaryDirectory() as tmpdir:
        for browser in browsers:
            for fmt in ["json3", "vtt", "srt"]:
                try:
                    output_path = os.path.join(tmpdir, "subtitle")
                    cmd = [
                        "yt-dlp", "--write-auto-sub", "--write-sub",
                        "--sub-lang", "en,en-US,en-GB",
                        "--sub-format", fmt, "--skip-download",
                        "--output", output_path, "--quiet",
                    ]
                    if browser:
                        cmd += ["--cookies-from-browser", browser]
                    cmd.append(video_url)
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                    sub_file = None
                    for fname in os.listdir(tmpdir):
                        if any(fname.endswith(e) for e in subtitle_extensions):
                            sub_file = os.path.join(tmpdir, fname)
                            break

                    if sub_file:
                        text = open(sub_file, encoding="utf-8").read()
                        text = re.sub(r"WEBVTT.*?\n|<[^>]+>|\d+\n[\d:,]+ --> [\d:,]+\n", "", text)
                        text = re.sub(r"\s+", " ", text).strip()
                        if len(text.split()) > 50:
                            return text
                except Exception:
                    continue

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        vid = extract_video_id(video_url)
        tl = YouTubeTranscriptApi.get_transcript(vid)
        return " ".join([e["text"] for e in tl]).strip()
    except Exception:
        pass

    return "Error: Could not fetch transcript."

def run_youtube_summarization(transcript: str):
    transcript = re.sub(r"\s+", " ", transcript).strip()
    word_count = len(transcript.split())
    if word_count < 80:
        st.error(f"Transcript too short ({word_count} words).")
        return
    st.success(f"✅ Transcript ready — {word_count} words")

    summary = summarize_full_text(transcript)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if len(s.strip()) > 30]
    sentences = [s if s[-1] in ".!?" else s+"." for s in sentences]

    overview   = sentences[0] if sentences else ""
    key_points = sentences[1:] if len(sentences) > 1 else sentences

    md  = "---\n## 📋 Video Overview\n\n"
    md += f"> {overview}\n\n"
    md += "---\n## 📌 Key Points\n\n"
    for pt in key_points:
        md += f"- {pt}\n"
    md += "\n---\n## 📊 Stats\n\n"
    md += f"| Transcript Words | {word_count} |\n|---|---|\n"
    md += f"| Key Points | {len(key_points)} |\n"
    md += "\n> 💡 *Tip: Read the key points aloud to memorize faster.*\n"

    st.divider()
    st.subheader("🎓 Your Study Notes")
    st.markdown(md)
    st.download_button("⬇️ Download Study Notes (.txt)", data=md,
                       file_name="study_notes.txt", mime="text/plain")

# ─────────────────────────────────────────────────────────────────────────────
# 🔟  UI
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["▶️  YouTube URL", "📁  Upload Video File"])

with tab1:
    st.markdown("#### Paste a YouTube video link")
    st.info("✅ Works best with videos that have a CC (captions) button on YouTube", icon="ℹ️")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
                        label_visibility="collapsed")
    if st.button("🚀 Generate Summary", key="yt_btn"):
        if not url:
            st.warning("Please enter a YouTube URL.")
        else:
            vid = extract_video_id(url)
            if not vid:
                st.error("❌ Invalid YouTube URL.")
            else:
                with st.spinner("📥 Fetching transcript..."):
                    transcript = fetch_transcript_youtube(url)
                if transcript.startswith("Error"):
                    st.error(transcript)
                else:
                    run_youtube_summarization(transcript)

with tab2:
    st.markdown("#### Upload your video file")
    st.info("🎬 Your video will be transcribed → summarized → turned into a short summary video!", icon="ℹ️")
    st.warning("⏱️ A 13-min video takes ~5–8 mins total. Please wait — don't close the app.")

    uploaded_file = st.file_uploader(
        "Choose a video (MP4, MKV, AVI, MOV, WEBM)",
        type=["mp4", "mkv", "avi", "mov", "webm"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.video(uploaded_file)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"📂 `{uploaded_file.name}` — {file_size_mb:.1f} MB")

        if file_size_mb > 500:
            st.error("❌ File too large. Please upload a video under 500 MB.")
        else:
            if st.button("🚀 Transcribe, Summarize & Create Summary Video", key="upload_btn"):
                process_uploaded_video(uploaded_file.read(), uploaded_file.name)