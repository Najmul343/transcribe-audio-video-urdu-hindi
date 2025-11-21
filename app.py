import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
import tempfile
import os
import re
from groq import Groq
from fpdf import FPDF  # Use fpdf2 for better Unicode
import requests
import base64
from io import BytesIO

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

# ------------------- ENHANCED LLM CORRECTION (RESTRICTED PROMPT) -------------------
def correct_urdu_chunk(raw_chunk: str) -> str:
    """
    Uses moonshotai/kimi-k2-instruct-0905 with restricted prompt: No major changes/paraphrasing.
    Processes 10–15 line chunks for context.
    """
    prompt = f"""Fix Urdu grammar and spelling in this speech-to-text transcript chunk.

CRITICAL RESTRICTIONS (MUST FOLLOW):
- DO NOT make major changes to sentences or paraphrase—keep original wording and structure intact.
- DO NOT add, remove, or alter any words unless it's an obvious spelling/grammar error (e.g., 'بیارات' → 'بیماریاں', 'فرون' → 'فرعون', 'توحید' → 'تواصی بالحق').
- Only fix obvious speech errors, add proper punctuation (، ؟ ۔ !), and ensure natural sentence flow.
- For Quranic/Hadith terms: Correct to standard forms (e.g., 'ماسلا رکھو' → 'مثل راکب') without changing meaning.
- Output ONLY the corrected text. No explanations, no extras.

Raw Chunk:
{raw_chunk}

Corrected Chunk:"""

    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",  # Your tested model—excellent for Urdu
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low for consistency
            max_tokens=1024,  # Enough for chunk
            top_p=0.9
        )
        corrected = response.choices[0].message.content.strip()
        # Extract only the corrected part (post-prompt)
        if "Corrected Chunk:" in corrected:
            corrected = corrected.split("Corrected Chunk:")[-1].strip()
        return corrected
    except Exception as e:
        st.error(f"LLM correction failed: {e}")
        return raw_chunk

# ------------------- MAIN APP LOGIC -------------------
uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp3","wav","m4a","mp4","mov","mkv"])

if st.button("✨ Generate Perfect Urdu Script", type="primary"):
    if not uploaded_file:
        st.error("Please upload an audio/video file")
        st.stop()

    # Step 1: Transcribe with Whisper
    with st.spinner("Transcribing audio with Whisper..."):
        placeholder = st.empty()
        placeholder.markdown("<div class='card'><h3 style='text-align:center'>Extracting & Transcribing...</h3></div>", unsafe_allow_html=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        model = WhisperModel("small", device="cpu", compute_type="int8")
        audio = decode_audio(temp_path)
        audio = np.array(audio).astype("float32")
        segments, _ = model.transcribe(audio, language="ur", vad_filter=True)

        raw_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
        os.unlink(temp_path)
        placeholder.empty()

    # Step 2: Chunk raw text (10–15 lines/chunk for context)
    sentences = re.split(r'(?<=[۔؟!])\s+', raw_text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len((current_chunk + " " + sent).split()) < 180:  # ~10–15 lines
            current_chunk += " " + sent
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sent
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Step 3: Correct each chunk with restricted LLM
    with st.spinner(f"AI Correcting Grammar & Spelling ({len(chunks)} chunks)..."):
        corrected_chunks = []
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            corrected = correct_urdu_chunk(chunk)
            corrected_chunks.append(corrected)
            progress_bar.progress((i + 1) / len(chunks))

    perfect_urdu = "\n\n".join(corrected_chunks).strip()
    # Light post-process: Ensure paragraph breaks after punctuation
    perfect_urdu = re.sub(r'([؟۔!])\s*([ا-ی][^۔؟!]{40,})', r'\1\n\n\2', perfect_urdu)

    st.balloons()
    st.success("✅ Perfect Urdu Script Generated! (Grammar & Spelling Fixed, No Paraphrasing)")

    # ------------------- BEAUTIFUL DISPLAY -------------------
    st.markdown(f"<div class='card'><div class='urdu'>{perfect_urdu}</div></div>", unsafe_allow_html=True)

    # ------------------- DOWNLOADS -------------------
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download TXT", perfect_urdu, "perfect_urdu.txt", "text/plain")
    with col2:
        try:
            # Download & embed Nastaliq font
            font_url = "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu-Regular.ttf"
            font_response = requests.get(font_url)
            font_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf").name
            with open(font_path, "wb") as f:
                f.write(font_response.content)

            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("NastaliqUrdu", "", font_path, uni=True)
            pdf.set_font("NastaliqUrdu", size=18)
            pdf.set_right_margin(20)  # RTL support
            pdf.multi_cell(0, 10, perfect_urdu)

            pdf_bytes = BytesIO()
            pdf.output(pdf_bytes)
            pdf_data = pdf_bytes.getvalue()

            os.unlink(font_path)  # Cleanup

            st.download_button("📄 Download PDF (Nastaliq Font)", pdf_data, "perfect_urdu.pdf", "application/pdf")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}. TXT download is perfect!")

st.caption("Powered by faster-whisper + Groq MoonshotAI Kimi-K2-Instruct-0905 • Precise Urdu Corrections (No Paraphrasing)")
