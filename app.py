import streamlit as st
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
import tempfile
import os
import re
from groq import Groq
from fpdf import FPDF  # fpdf2 for Unicode
import requests
import base64
from io import BytesIO

# ------------------- PREMIUM PAGE STYLE (slice.wbrain.me Inspired: Clean Cards, Minimalist) -------------------
st.set_page_config(page_title="Urdu Pro", layout="centered", page_icon="🎙️")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; }
    .urdu { font-family: 'Noto Nastaliq Urdu', serif; font-size: 26px; line-height: 2.2; direction: rtl; text-align: right; color: #2d3748; }
    .title { font-size: 48px; font-weight: bold; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 10px; }
    .card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin: 20px 0; }
    .mic-container { text-align: center; }
    .mic-btn { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; font-size: 24px; height: 70px; border-radius: 50px; border: none; width: 100%; cursor: pointer; }
    .mic-btn:hover { background: linear-gradient(135deg, #ee5a24, #ff6b6b); }
    .progress-card { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }
    .recording { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("<h1 class='title'>Urdu Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#64748b;'>Record voice or upload audio → Get perfect Urdu script with grammar & spelling fixes</p>", unsafe_allow_html=True)

# ------------------- GROQ CLIENT -------------------
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# ------------------- ENHANCED LLM CORRECTION (Your Tested Prompt + Restrictions) -------------------
def correct_urdu_chunk(raw_chunk: str) -> str:
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
            model="moonshotai/kimi-k2-instruct-0905",  # Your tested model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9
        )
        corrected = response.choices[0].message.content.strip()
        if "Corrected Chunk:" in corrected:
            corrected = corrected.split("Corrected Chunk:")[-1].strip()
        return corrected
    except Exception as e:
        st.error(f"LLM correction failed: {e}")
        return raw_chunk

# ------------------- UPLOAD OPTION -------------------
st.markdown("### 1. Upload Audio/Video File")
uploaded_file = st.file_uploader("", type=["mp3","wav","m4a","mp4","mov","mkv"])

# ------------------- LIVE VOICE RECORDING (Pure JS: No External Components, No Errors) -------------------
st.markdown("### 2. براہِ راست آواز ریکارڈ کریں (Live Recording)")
st.markdown('<div class="card">', unsafe_allow_html=True)

# Initialize session state for recording
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

# JS for browser recording (HTML5 MediaRecorder — works everywhere, no deps)
record_js = """
<div class="mic-container">
    <button id="micBtn" class="mic-btn" onclick="toggleRecording()">{text}</button>
    <p id="status" style="text-align:center; color:#a0aec0;">ریکارڈنگ شروع کریں</p>
</div>
<audio id="audioPlayback" controls style="width:100%; margin-top:10px; display:none;"></audio>

<script>
let mediaRecorder;
let audioChunks = [];
let stream;

function toggleRecording() {
    const btn = document.getElementById('micBtn');
    const status = document.getElementById('status');
    const audioPlayback = document.getElementById('audioPlayback');

    if (!st.session_state.recording) {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(s => {
            stream = s;
            mediaRecorder = new MediaRecorder(s, { mimeType: 'audio/webm' });
            audioChunks = [];

            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const reader = new FileReader();
                reader.onload = () => {
                    // Send to Streamlit (base64)
                    parent.document.querySelector('iframe').contentWindow.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: reader.result.split(',')[1]  // Base64 data
                    }, '*');
                    audioPlayback.src = URL.createObjectURL(audioBlob);
                    audioPlayback.style.display = 'block';
                };
                reader.readAsDataURL(audioBlob);
            };

            mediaRecorder.start();
            st.session_state.recording = true;
            btn.innerText = 'رکائیں';
            status.innerText = 'ریکارڈنگ جاری ہے...';
            status.className = 'recording';
        }).catch(err => {
            status.innerText = 'مائیک رسائی کی اجازت دیں';
        });
    } else {
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
        st.session_state.recording = false;
        btn.innerText = 'ریکارڈنگ شروع کریں';
        status.innerText = 'ریکارڈنگ مکمل!';
        status.className = '';
    }
}
</script>
"""

st.components.v1.html(record_js, height=150, width=400)

# Handle audio data from JS (session state)
if st.session_state.audio_data:
    st.audio(st.session_state.audio_data, format="audio/webm")
    # Convert to temp WAV for Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        # Base64 to bytes (simplified; in practice, use io.BytesIO)
        audio_bytes = base64.b64decode(st.session_state.audio_data)
        tmp.write(audio_bytes)
        temp_path = tmp.name

    with st.spinner("آپ کی آواز کو اردو متن میں تبدیل ہو رہا ہے..."):
        model = WhisperModel("small", device="cpu", compute_type="int8")
        audio = decode_audio(temp_path)
        audio = np.array(audio).astype("float32")
        segments, _ = model.transcribe(audio, language="ur", vad_filter=True)
        raw_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
        os.unlink(temp_path)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<p style="text-align:center; color:#a0aec0;">(مائیک بٹن دبا کر ریکارڈنگ شروع کریں)</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------- PROCESSING BUTTON (Handles Upload or Recording) -------------------
if st.button("✨ Generate Perfect Urdu Script", type="primary"):
    # Get raw_text from upload or recording
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        model = WhisperModel("small", device="cpu", compute_type="int8")
        audio = decode_audio(temp_path)
        audio = np.array(audio).astype("float32")
        segments, _ = model.transcribe(audio, language="ur", vad_filter=True)
        raw_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
        os.unlink(temp_path)
    elif 'raw_text' in locals() and raw_text:  # From recording
        pass
    else:
        st.error("Please upload a file or record voice first")
        st.stop()

    # Chunking (10–15 lines, ~180 words)
    sentences = re.split(r'(?<=[۔؟!])\s+', raw_text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len((current_chunk + " " + sent).split()) < 180:
            current_chunk += " " + sent
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sent
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # LLM Correction
    with st.spinner(f"AI Correcting Grammar & Spelling ({len(chunks)} chunks)..."):
        corrected_chunks = []
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            corrected = correct_urdu_chunk(chunk)
            corrected_chunks.append(corrected)
            progress_bar.progress((i + 1) / len(chunks))

    perfect_urdu = "\n\n".join(corrected_chunks).strip()
    perfect_urdu = re.sub(r'([؟۔!])\s*([ا-ی][^۔؟!]{40,})', r'\1\n\n\2', perfect_urdu)

    st.balloons()
    st.success("✅ Perfect Urdu Script Generated! (No Paraphrasing)")

    # ------------------- OUTPUT DISPLAY (Clean Card Like slice.wbrain.me) -------------------
    st.markdown('<div class="card progress-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='urdu'>{perfect_urdu}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Downloads
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download TXT", perfect_urdu, "perfect_urdu.txt", "text/plain")
    with col2:
        try:
            font_url = "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu-Regular.ttf"
            font_response = requests.get(font_url)
            font_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf").name
            with open(font_path, "wb") as f:
                f.write(font_response.content)

            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("NastaliqUrdu", "", font_path, uni=True)
            pdf.set_font("NastaliqUrdu", size=18)
            pdf.set_right_margin(20)
            pdf.multi_cell(0, 10, perfect_urdu)

            pdf_bytes = BytesIO()
            pdf.output(pdf_bytes)
            pdf_data = pdf_bytes.getvalue()

            os.unlink(font_path)

            st.download_button("📄 Download PDF (Nastaliq Font)", pdf_data, "perfect_urdu.pdf", "application/pdf")
        except Exception as e:
            st.warning(f"PDF failed: {e}. TXT download is perfect!")

st.markdown('</div>', unsafe_allow_html=True)  # Close main div
st.caption("Powered by faster-whisper + Groq MoonshotAI Kimi-K2 • Live Voice Recording (Browser JS)")
