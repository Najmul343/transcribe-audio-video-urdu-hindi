import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
import tempfile
import os
import re
from groq import Groq
from fpdf import FPDF
import base64
from io import BytesIO

# ------------------- PREMIUM PAGE STYLE -------------------
st.set_page_config(page_title="Urdu Pro", layout="centered", page_icon="black_circle")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    .urdu { font-family: 'Noto Nastaliq Urdu', serif; font-size: 28px; line-height: 2.3; direction: rtl; text-align: right; color: #1e293b; }
    .title { font-size: 52px; font-weight: bold; background: linear-gradient(90deg, #1e40af, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .card { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .stButton>button { background: #1e40af; color: white; font-size: 20px; height: 60px; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Urdu Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#64748b;'>Upload audio → Get perfect newspaper-quality Urdu script instantly</p>", unsafe_allow_html=True)

# ------------------- GROQ CLIENT -------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ------------------- UPLOAD & LANGUAGE -------------------
uploaded_file = st.file_uploader("", type=["mp3","wav","m4a","mp4","mov","mkv"])
language_choice = "ur"  # Forced Urdu

if st.button("✨ Generate Perfect Urdu Script", type="primary"):
    if not uploaded_file:
        st.error("Please upload an audio/video file")
        st.stop()

    # Premium loading
    with st.spinner(""):
        placeholder = st.empty()
        placeholder.markdown("<div class='card'><h2 style='text-align:center'>Processing your audio…</h2></div>", unsafe_allow_html=True)

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Whisper transcription
        model = WhisperModel("small", device="cpu", compute_type="int8")
        audio = decode_audio(temp_path)
        audio = np.array(audio).astype("float32")
        segments, _ = model.transcribe(audio, language="ur", vad_filter=True)

        raw = " ".join(seg.text.strip() for seg in segments)

        # AI Correction (exact words, perfect punctuation & paragraphs)
        prompt = f"""Fix ONLY spelling, punctuation, spacing and paragraph breaks.
Keep every single word exactly the same. No paraphrasing, no additions, no removals.

RAW:
{raw}

PERFECT URDU SCRIPT:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096
        )
        perfect_urdu = response.choices[0].message.content.strip()
        if "PERFECT URDU SCRIPT:" in perfect_urdu:
            perfect_urdu = perfect_urdu.split("PERFECT URDU SCRIPT:")[-1].strip()

        # Clean up temp file
        os.unlink(temp_path)

        # Final beautiful display
        placeholder.empty()
        st.balloons()

        st.markdown(f"<div class='card'><div class='urdu'>{perfect_urdu}</div></div>", unsafe_allow_html=True)

        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download TXT", perfect_urdu, "perfect_urdu.txt", "text/plain")
        with col2:
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("NotoNastaliq", "", "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf", uni=True)
            pdf.set_font("NotoNastaliq", size=20)
            pdf.multi_cell(0, 12, perfect_urdu)
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            b64 = base64.b64encode(pdf_bytes).decode()
            st.download_button("Download PDF", pdf_bytes, "perfect_urdu.pdf", "application/pdf")

st.caption("Powered by AI-LLMs • Made for perfect Urdu")
