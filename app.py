# app.py - Urdu Live Transcriber + Auto Corrector (Streamlit App)

import streamlit as st
from faster_whisper import WhisperModel
import torch
from pydub import AudioSegment
import os
import tempfile
import re

# ---------- Page Config ----------
st.set_page_config(page_title="اردو آڈیو → خوبصورت متن", page_icon="Pakistan", layout="centered")

st.title("Pakistan اردو آڈیو ٹرانسکرائبر + خودکار اصلاح")
st.markdown("**مفت، تیز، بہترین Urdu Whisper + Spell Correction** – 2025 ایڈیشن")

# ---------- Load Models (cached) ----------
@st.cache_resource
def load_whisper_model():
    with st.spinner("Whisper ماڈل لوڈ ہو رہا ہے (صرف ایک بار)..."):
        return WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

@st.cache_resource
def load_corrector():
    with st.spinner("ہجے درست کرنے والا ماڈل لوڈ ہو رہا ہے..."):
        from transformers import pipeline
        return pipeline("text2text-generation", 
                        model="oliverguhr/spelling-correction-multilingual-base",
                        device=0 if torch.cuda.is_available() else -1)

model = load_whisper_model()
corrector = load_corrector()

# ---------- File Upload ----------
uploaded_file = st.file_uploader("اپنی آڈیو فائل ڈالیں (mp3, m4a, wav, ogg, mp4 وغیرہ)", 
                                  type=["mp3", "m4a", "wav", "ogg", "mp4", "webm"])

if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    st.audio(audio_path, format="audio/wav")
    st.info("فائل لوڈ ہو گئی – ٹرانسکریپشن شروع...")

    # Convert to WAV
    with st.spinner("آڈیو کو تیار کیا جا رہا ہے..."):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_path = "/tmp/streamlit_input.wav"
        audio.export(wav_path, format="wav")

    # Transcribe with Urdu Whisper
    with st.spinner("اردو میں ٹرانسکریپٹ ہو رہا ہے (large-v3 ماڈل)..."):
        segments, info = model.transcribe(wav_path, language="ur", vad_filter=True, beam_size=7)
        raw_text = " ".join([seg.text.strip() for seg in segments])

    st.success(f"زبان شناخت ہوئی: اردو ({info.language_probability:.1%} یقین)")

    # Show raw transcription
    st.subheader("خام ٹرانسکریپشن (Whisper سے)")
    st.text_area("خام", raw_text, height=150)

    # Auto-correct with pretrained model
    with st.spinner("ہجے، گرامر، وقفے درست کیے جا رہے ہیں... (تیز ترین AI)"):
        prompt = f"correct spelling and grammar in Urdu: {raw_text}"
        corrected = corrector(prompt, max_new_tokens=2048, temperature=0.0, do_sample=False)[0]['generated_text']
        final_text = corrected.replace("correct spelling and grammar in Urdu:", "").strip()

        # Final polish
        final_text = re.sub(r'\s+', ' ', final_text)
        final_text = re.sub(r' ([،؛؟!۔])', r'\1', final_text)
        final_text = re.sub(r'(\S)([۔؟!])', r'\1\2 ', final_text).strip()

    # Final Beautiful Output
    st.subheader("Pakistan خوبصورت، درست شدہ اردو متن")
    st.success("تیار ہے! کاپی کریں اور استعمال کریں")
    st.markdown(f"<div style='font-size:18px; line-height:1.8; direction:rtl; text-align:right;'>{final_text}</div>", 
                unsafe_allow_html=True)

    # Download button
    st.download_button("متن ڈاؤن لوڈ کریں (.txt)", final_text, file_name="اردو_ٹرانسکریپشن.txt")

    # Copy to clipboard button
    st.code(f"navigator.clipboard.writeText(`{final_text}`)", language="javascript")

    # Clean up
    os.unlink(audio_path)
    if os.path.exists(wav_path):
        os.unlink(wav_path)

else:
    st.info("Pakistan اوپر فائل اپ لوڈ کریں – mp3, WhatsApp voice note, YouTube clip وغیرہ سب چلیں گے!")

st.markdown("---")
st.caption("بنایا گیا پاکستانیوں کے لیے Pakistan • faster-whisper + T5-multilingual • 2025")
