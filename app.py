import streamlit as st
import os, yt_dlp, requests
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pathlib import Path

# Optional transliteration
try:
    from aksharamukha.transliterate import process as akshara_convert
except:
    akshara_convert = None

st.set_page_config(page_title="Whisper Transcriber", layout="wide")
st.title("🎙️ Whisper Transcriber (YouTube + Upload)")
st.write("Select language → Upload a file OR paste a YouTube link → Transcribe.")

# -----------------------------------------------------
# INPUT MODE
# -----------------------------------------------------
mode = st.radio("Choose Input Type:", ["Upload File", "YouTube URL"], horizontal=True)

youtube_url = ""
uploaded_file = None

if mode == "YouTube URL":
    youtube_url = st.text_input("Paste YouTube Link Here:")

if mode == "Upload File":
    uploaded_file = st.file_uploader("Upload Audio/Video File",
                                     type=["mp3","wav","m4a","mp4","mov"])


# -----------------------------------------------------
# LANGUAGE SELECTOR
# -----------------------------------------------------
language_choice = st.selectbox(
    "Select transcription language:",
    [
        "Urdu (Arabic Script)",
        "Hindi (Devanagari Script)",
        "Roman (Hinglish)",
        "English"
    ],
)

# -----------------------------------------------------
# DOWNLOAD (Simplified for Streamlit Cloud)
# -----------------------------------------------------
def download_youtube(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "yt_audio.%(ext)s",
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for f in os.listdir():
            if f.startswith("yt_audio") and f.endswith(".mp3"):
                return f

    except Exception as e:
        st.error("YouTube download failed. Try a different link.")
        return None

    return None


# Transliteration helpers
def to_hindi(text):
    if akshara_convert:
        try:
            return akshara_convert("ISO", "Devanagari", text)
        except:
            return text
    return text

def to_urdu(text):
    if akshara_convert:
        try:
            return akshara_convert("ISO", "Arabic", text)
        except:
            return text
    return text


# -----------------------------------------------------
# TRANSCRIBE BUTTON
# -----------------------------------------------------
if st.button("Start Transcription"):
    st.write("### ⏳ Processing...")

    # 🎬 Funny GIF
    gif_placeholder = st.empty()
    gif_placeholder.image(
        "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW5yeTl3OW0weG1sbnMydGhycjY0a2I2ZGlyNWQwazVzaTYycnMzMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jU8yHwHSAoHvepN65t/giphy.gif",
        width=350
    )
    st.write("👨‍💻 Transcribing your audio... He's writing FAST! 💨")

    # -----------------------------------------------------
    # LOAD INPUT
    # -----------------------------------------------------
    if mode == "Upload File":
        if uploaded_file is None:
            st.error("❌ Please upload a file first.")
            gif_placeholder.empty()
            st.stop()

        with open("inputfile", "wb") as f:
            f.write(uploaded_file.getbuffer())
        filename = "inputfile"

    else:
        if youtube_url.strip() == "":
            st.error("❌ Please paste a YouTube URL.")
            gif_placeholder.empty()
            st.stop()

        filename = download_youtube(youtube_url)
        if filename is None:
            gif_placeholder.empty()
            st.stop()

    # Convert to WAV
    audio = AudioSegment.from_file(filename)
    audio.export("audio.wav", format="wav")

    # -----------------------------------------------------
    # LOAD WHISPER (CPU MODE)
    # -----------------------------------------------------
    st.write("🔧 Loading Whisper model (small) — CPU mode for Streamlit Cloud...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    lang_map = {
        "Urdu (Arabic Script)": "ur",
        "Hindi (Devanagari Script)": "hi",
        "Roman (Hinglish)": "en",
        "English": "en"
    }

    lang_code = lang_map[language_choice]

    # -----------------------------------------------------
    # TRANSCRIPTION
    # -----------------------------------------------------
    segments, _ = model.transcribe(
        "audio.wav",
        language=lang_code,
        vad_filter=True
    )

    final_text = ""

    for seg in segments:
        text = seg.text.strip()

        if language_choice == "Hindi (Devanagari Script)":
            text = to_hindi(text)
        elif language_choice == "Urdu (Arabic Script)":
            text = to_urdu(text)

        final_text += text + " "

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------
    st.success("✅ Transcription Completed Successfully!")

    gif_placeholder.empty()

    st.text_area("📄 Transcription Output:", final_text, height=300)

    st.download_button(
        "⬇ Download Transcript",
        data=final_text,
        file_name="transcript.txt"
    )
