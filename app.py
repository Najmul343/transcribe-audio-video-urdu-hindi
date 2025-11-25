import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Urdu Transcriber",
    page_icon="📝",
    layout="centered"
)

st.title("🇵🇰 Urdu Audio/Video Transcriber")
st.markdown("### WhatsApp voice → Beautiful Urdu Text")
st.caption("2025 • CPU Optimized • No Crashes")

# ---------------------------------------------------------
# Load Whisper Model (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = WhisperModel(
        "tiny",                # safest for cloud, fast
        device="cpu",
        compute_type="int8"
    )
    return model

model = load_model()
st.success("Model Loaded Successfully ✔️")

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
file = st.file_uploader(
    "Upload Audio or Video File",
    type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"]
)

# ---------------------------------------------------------
# Process File
# ---------------------------------------------------------
if file:
    st.audio(file)

    if st.button("Transcribe to Urdu"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        with st.spinner("Transcribing... Please wait ⏳"):
            segments, info = model.transcribe(
                temp_path,
                language="ur",
                beam_size=5
            )
            output_text = " ".join([seg.text for seg in segments])

        os.remove(temp_path)

        st.success("Transcription Complete ✔️")
        st.markdown("### Urdu Transcript:")
        st.write(output_text)

        st.download_button(
            label="Download Urdu Text",
            data=output_text,
            file_name="transcript_urdu.txt"
        )

        st.balloons()
else:
    st.info("Please upload an audio/video file.")
