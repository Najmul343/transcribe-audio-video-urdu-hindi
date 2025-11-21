import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
import tempfile
import os
import re
from groq import Groq
from fpdf import FPDF  # Now fpdf2
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

# ------------------- UPLOAD & LANGUAGE -------------------
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

    # ------------------- ENHANCED LLM CORRECTION (Sentence-by-Sentence) -------------------
    with st.spinner("AI Processing: Correcting grammar, spelling & flow word-by-word..."):
        corrected_sentences = []
        for sent in raw_sentences:
            if len(sent) < 3:  # Skip noise
                continue
            # Enhanced prompt: Focus on each word/sentence for sense-making corrections
            prompt = f"""You are an expert Urdu grammarian and editor. Process this single sentence from speech-to-text.

RULES (STRICT):
- Analyze EACH WORD for spelling/grammar errors and correct ONLY if it makes grammatical/semantic sense (e.g., 'بیارات' → 'بیماری' if context fits; keep original if ambiguous).
- Fix sentence structure for natural flow, but DO NOT change meaning, add/remove words, or paraphrase.
- Add proper Urdu punctuation (، ؟ ۔) and ensure readability.
- Output ONLY the corrected sentence. No explanations.

RAW SENTENCE:
{sent}

CORRECTED SENTENCE:"""

            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Better multilingual/Urdu model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,  # Ultra-low for precision
                max_tokens=256  # Per sentence
            )
            corrected_sent = response.choices[0].message.content.strip()
            if "CORRECTED SENTENCE:" in corrected_sent:
                corrected_sent = corrected_sent.split("CORRECTED SENTENCE:")[-1].strip()
            corrected_sentences.append(corrected_sent)

        perfect_urdu = ' '.join(corrected_sentences).strip()
        # Add paragraph breaks heuristically (after questions/topics)
        perfect_urdu = re.sub(r'([؟۔])\s*([ا-ی][^۔؟]{50,})', r'\1\n\n\2', perfect_urdu)

    st.balloons()
    st.success("✅ Perfect Urdu Script Generated!")

    # ------------------- BEAUTIFUL DISPLAY -------------------
    st.markdown(f"<div class='card'><div class='urdu'>{perfect_urdu}</div></div>", unsafe_allow_html=True)

    # ------------------- DOWNLOADS -------------------
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download TXT", perfect_urdu, "perfect_urdu.txt", "text/plain")
    with col2:
        # Fixed PDF Generation with fpdf2 + Downloaded Font
        try:
            # Download Noto Nastaliq Urdu TTF (free Google font)
            font_url = "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu-Regular.ttf"
            font_response = requests.get(font_url)
            font_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf").name
            with open(font_path, "wb") as f:
                f.write(font_response.content)

            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("NastaliqUrdu", "", font_path, uni=True)  # Embed local TTF
            pdf.set_font("NastaliqUrdu", size=18)
            pdf.set_right_margin(20)  # RTL support
            pdf.multi_cell(0, 10, perfect_urdu)  # Line height for readability

            pdf_bytes = BytesIO()
            pdf.output(pdf_bytes)
            pdf_data = pdf_bytes.getvalue()

            # Clean temp font file
            os.unlink(font_path)

            st.download_button("📄 Download PDF (Nastaliq Font)", pdf_data, "perfect_urdu.pdf", "application/pdf")
        except Exception as e:
            st.warning(f"PDF failed (font issue): {e}. TXT download is perfect!")

st.caption("Powered by faster-whisper + Groq Llama 3.1 70B • Grammar & Spelling Perfected")
