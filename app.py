%%writefile app.py
import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
import tempfile

# Optional transliteration
try:
    from aksharamukha.transliterate import process as akshara_convert
except:
    akshara_convert = None

# -----------------------------------------------
# PAGE UI
# -----------------------------------------------
st.set_page_config(page_title="Whisper Transcriber", layout="wide")
st.title("🎙️ Whisper Transcriber (Upload Audio/Video)")
st.write("Upload audio/video → select language → get transcript.")

# -----------------------------------------------
# FILE UPLOADER
# -----------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Audio or Video File",
    type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"]
)

# -----------------------------------------------
# LANGUAGE SELECTOR
# -----------------------------------------------
language_choice = st.selectbox(
    "Select transcription language:",
    ["Urdu (Arabic Script)",
     "Hindi (Devanagari Script)",
     "Roman (Hinglish)",
     "English"]
)

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

# -----------------------------------------------
# TRANSCRIBE BUTTON
# -----------------------------------------------
if st.button("Start Transcription"):
    if not uploaded_file:
        st.error("Please upload a file first!")
        st.stop()

    # GIF while busy
    gif_placeholder = st.empty()
    gif_placeholder.image(
        "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW5yeTl3OW0weG1sbnMydGhycjY0a2I2ZGlyNWQwazVzaTYycnMzMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jU8yHwHSAoHvepN65t/giphy.gif",
        width=350
    )
    st.write("👨‍💻 Extracting audio & transcribing...")

    # -----------------------------------------------
    # SAVE UPLOADED FILE TEMPORARILY
    # -----------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # -----------------------------------------------
    # LOAD WHISPER MODEL (CPU SAFE)
    # -----------------------------------------------
    model = WhisperModel("small", device="cpu", compute_type="int8")

    # -----------------------------------------------
    # EXTRACT AUDIO DIRECTLY (NO FFMPEG NEEDED)
    # -----------------------------------------------
    try:
        # decode_audio returns 16kHz floating audio array
        audio = decode_audio(temp_path)
    except Exception as e:
        gif_placeholder.empty()
        st.error(f"Audio extraction failed: {e}")
        st.stop()

    # Convert to Whisper’s expected format
    audio = np.array(audio).astype("float32")

    lang_map = {
        "Urdu (Arabic Script)": "ur",
        "Hindi (Devanagari Script)": "hi",
        "Roman (Hinglish)": "en",
        "English": "en"
    }
    lang_code = lang_map[language_choice]

    # -----------------------------------------------
    # TRANSCRIBE
    # -----------------------------------------------
    segments, _ = model.transcribe(
        audio,
        language=lang_code,
        vad_filter=True
    )

    final_text = ""
    for seg in segments:
        t = seg.text.strip()

        if language_choice == "Hindi (Devanagari Script)":
            t = to_hindi(t)
        elif language_choice == "Urdu (Arabic Script)":
            t = to_urdu(t)

        final_text += t + " "

    # -----------------------------------------------
    # OUTPUT
    # -----------------------------------------------
    gif_placeholder.empty()
    st.success("✅ Transcription Completed!")

    st.text_area("📄 Transcription Output:", final_text, height=300)

    st.download_button(
        "⬇ Download Transcript",
        data=final_text,
        file_name="transcript.txt"
    )
