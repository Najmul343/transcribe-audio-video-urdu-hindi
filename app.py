import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
import tempfile
import os
import re
from groq import Groq
from fpdf import FPDF  # fpdf2
import base64
from io import BytesIO
import requests  # For font download

# ------------------- PREMIUM PAGE STYLE -------------------
st.set_page_config(page_title="Urdu Pro", layout="centered", page_icon="🎙️")
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
st.markdown("<p style='text-align:center; font-size:18px; color:#64748b;'>Upload audio → Get perfect, grammatically correct Urdu script instantly</p>", unsafe_allow_html=True)

# ------------------- GROQ CLIENT -------------------
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# ------------------- UPLOAD -------------------
uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp3","wav","m4a","mp4","mov","mkv"])
if st.button("✨ Generate Perfect Urdu Script", type="primary"):
    if not uploaded_file:
        st.error("Please upload an audio/video file")
        st.stop()

    # Premium loading
    with st.spinner("Processing your audio…"):
        placeholder = st.empty()
        placeholder.markdown("<div class='card'><h2 style='text-align:center'>Extracting & Transcribing...</h2></div>", unsafe_allow_html=True)

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Whisper transcription
        model = WhisperModel("small", device="cpu", compute_type="int8")
        audio = decode_audio(temp_path)
        audio = np.array(audio).astype("float32")
        segments, _ = model.transcribe(audio, language="ur", vad_filter=True)

        raw_sentences = [seg.text.strip() for seg in segments if seg.text.strip()]

        # Clean up temp file
        os.unlink(temp_path)

        placeholder.empty()

    # ------------------- ENHANCED LLM CORRECTION (Word/Sentence-by-Sentence) -------------------
    with st.spinner("AI Processing: Scanning each word & sentence for grammar/spelling corrections..."):
        corrected_sentences = []
        batch_size = 2  # Process 1-2 sentences per call for precision
        for i in range(0, len(raw_sentences), batch_size):
            batch = ' '.join(raw_sentences[i:i+batch_size])
            if len(batch) < 5:  # Skip noise
                continue
            # Enhanced prompt: Word-by-word + sentence sense-making
            prompt = f"""You are an expert Urdu grammarian. Process this batch of 1-2 sentences from speech-to-text.

STRICT RULES:
- Scan EACH WORD individually: Correct spelling/grammar ONLY if it makes semantic/grammatical sense in context (e.g., 'بیارات' → 'بیماری' for disease; 'جاہداد' → 'جائیداد' for property; keep ambiguous words original).
- For each SENTENCE: Ensure natural flow and structure without changing meaning, adding/removing words, or paraphrasing.
- Add proper Urdu punctuation (، ؟ ۔ !) and fix spacing for readability.
- Output ONLY the corrected sentences, separated by spaces. No explanations or extras.

RAW BATCH:
{batch}

CORRECTED BATCH:"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Valid & fast Groq model (fixed error)
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,  # Low for precise corrections
                max_tokens=512  # Enough for batch
            )
            corrected_batch = response.choices[0].message.content.strip()
            if "CORRECTED BATCH:" in corrected_batch:
                corrected_batch = corrected_batch.split("CORRECTED BATCH:")[-1].strip()
            corrected_sentences.extend(corrected_batch.split(' ') if ' ' in corrected_batch else [corrected_batch])

        perfect_urdu = ' '.join(corrected_sentences).strip()
        # Smart paragraph breaks: After questions or long pauses
        perfect_urdu = re.sub(r'([؟۔!])\s*([ا-ی][^۔؟!]{40,})', r'\1\n\n\2', perfect_urdu)

    st.balloons()
    st.success("✅ Perfect Urdu Script Generated! (Word-by-word grammar & spelling fixed)")

    # ------------------- BEAUTIFUL DISPLAY -------------------
    st.markdown(f"<div class='card'><div class='urdu'>{perfect_urdu}</div></div>", unsafe_allow_html=True)

    # ------------------- DOWNLOADS -------------------
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download TXT", perfect_urdu, "perfect_urdu.txt", "text/plain")
    with col2:
        # Fixed PDF with downloaded font
        try:
            # Download Noto Nastaliq Urdu TTF
            font_url = "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu-Regular.ttf"
            font_response = requests.get(font_url)
            font_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf").name
            with open(font_path, "wb") as f:
                f.write(font_response.content)

            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("NastaliqUrdu", "", font_path, uni=True)
            pdf.set_font("NastaliqUrdu", size=18)
            pdf.set_right_margin(20)  # RTL
            pdf.multi_cell(0, 10, perfect_urdu)

            pdf_bytes = BytesIO()
            pdf.output(pdf_bytes)
            pdf_data = pdf_bytes.getvalue()

            os.unlink(font_path)  # Cleanup

            st.download_button("📄 Download PDF (Nastaliq Font)", pdf_data, "perfect_urdu.pdf", "application/pdf")
        except Exception as e:
            st.warning(f"PDF failed: {e}. TXT is perfect!")

st.caption("Powered by faster-whisper + Groq Llama 3.1 8B • Precise Word/Sentence Corrections")
