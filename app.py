# app.py
import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
import tempfile
import os
import re
from groq import Groq
from fpdf import FPDF
import requests
import base64
from io import BytesIO

# ------------------- PAGE STYLE -------------------
st.set_page_config(page_title="Urdu Pro", layout="centered", page_icon="mic")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    .urdu { font-family: 'Noto Nastaliq Urdu', serif; font-size: 28px; line-height: 2.4; direction: rtl; text-align: right; color: #1e293b; }
    .title { font-size: 52px; font-weight: bold; background: linear-gradient(90deg, #1e40af, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .card { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .stButton>button { background: #1e40af; color: white; font-size: 20px; height: 60px; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Urdu Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#64748b;'>Upload any audio → Get perfect, newspaper-quality Urdu script instantly</p>", unsafe_allow_html=True)

# ------------------- GROQ CLIENT -------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ------------------- CORRECTION FUNCTION (SUPER PROMPT) -------------------
def correct_urdu_chunk(raw_chunk: str) -> str:
    prompt = f"""تم ایک ماہرِ اردو ایڈیٹر اور اسلامی خطبات کے پروف ریڈر ہو۔  
نیچے دیا گیا متن ایک تقریر کا غلطِ املا، بغیرِ وقف اور شور والا ٹرانسکریپٹ ہے۔  

تمہارا کام ہے:  
- تمام واضح غلطیاں درست کرو (مثلاً فرون → فرعون، ابوزر → ابو ذر غفاری، بقمحصل → مقام حاصل، ماسلا رکھو → مثل راکب، توحید بل حق → تواصی بالحق، توحید بل صالح → تواصی بالصبر وغیرہ)  
- قرآن کی آیات اور احادیث کو بالکل درست لکھو  
- صحابہ کرام، انبیاء اور تاریخی نام درست کرو  
- جملوں کو مکمل کرو، صحیح وقف (۔ ، ؟ !) لگاؤ  
- قدرتی پیراگراف بناؤ  
- معنی یا الفاظ بالکل تبدیل مت کرو — صرف غلطیاں درست کرو  

خام متن:
{raw_chunk}

صرف درست شدہ، خوبصورت، مکمل پڑھنے کے قابل اردو متن آؤٹ پٹ کرو۔ کوئی وضاحت یا نوٹ نہیں:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # آپ کا پسندیدہ ماڈل
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"LLM error: {e}")
        return raw_chunk

# ------------------- MAIN APP -------------------
uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp3","wav","m4a","mp4","mov","mkv"])

if st.button("Generate Perfect Urdu Script", type="primary"):
    if not uploaded_file:
        st.error("Please upload a file")
        st.stop()

    with st.spinner("Transcribing with Whisper..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        model = WhisperModel("small", device="cpu", compute_type="int8")
        audio = decode_audio(temp_path)
        audio = np.array(audio).astype("float32")
        segments, _ = model.transcribe(audio, language="ur", vad_filter=True)
        raw_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
        os.unlink(temp_path)

    # Split into 10–15 line chunks (~150–200 words)
    sentences = re.split(r'(?<=[۔؟!])\s+', raw_text)
    chunks = []
    current = ""
    for s in sentences:
        if len((current + " " + s).split()) < 180:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())

    # Correct each chunk
    with st.spinner(f"Correcting grammar & spelling with AI ({len(chunks)} chunks)..."):
        corrected_chunks = []
        progress = st.progress(0)
        for i, chunk in enumerate(chunks):
            corrected = correct_urdu_chunk(chunk)
            corrected_chunks.append(corrected)
            progress.progress((i + 1) / len(chunks))

    perfect_urdu = "\n\n".join(corrected_chunks)

    st.balloons()
    st.success("Perfect Urdu Script Ready!")

    # Display
    st.markdown(f"<div class='card'><div class='urdu'>{perfect_urdu}</div></div>", unsafe_allow_html=True)

    # Downloads
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download TXT", perfect_urdu, "perfect_urdu.txt", "text/plain")
    with col2:
        try:
            font_url = "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu-Regular.ttf"
            font_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf").name
            with open(font_path, "wb") as f:
                f.write(requests.get(font_url).content)

            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("Nastaliq", "", font_path, uni=True)
            pdf.set_font("Nastaliq", size=18)
            pdf.multi_cell(0, 11, perfect_urdu)
            pdf_bytes = pdf.output(dest="S").encode("latin1")

            os.unlink(font_path)
            st.download_button("Download PDF", pdf_bytes, "perfect_urdu.pdf", "application/pdf")
        except:
            st.warning("PDF failed. Use TXT download.")

st.caption("Powered by faster-whisper + Groq Llama-3.1-8B-Instant • Best Urdu Bayan Correction 2025")
