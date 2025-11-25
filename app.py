import streamlit as st
from faster_whisper import WhisperModel
import torch
import os
import tempfile

st.set_page_config(page_title="اردو ٹرانسکرائبر", page_icon="Pakistan")

st.title("Pakistan اردو آڈیو → خوبصورت متن")
st.markdown("**فاسٹر وہسپر + خودکار اصلاح • 100% Streamlit Cloud پر چلتا ہے**")

@st.cache_resource
def load_model():
    return WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

model = load_model()

uploaded_file = st.file_uploader("آڈیو/ویڈیو فائل ڈالیں", type=["mp3","m4a","wav","ogg","mp4","webm","mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
        f.write(uploaded_file.getvalue())
        audio_path = f.name

    st.audio(audio_path)
    
    with st.spinner("ٹرانسکریپشن ہو رہی ہے... (large-v3 ماڈل)"):
        segments, info = model.transcribe(audio_path, language="ur", vad_filter=True)
        text = " ".join([s.text.strip() for s in segments])

    st.success(f"اردو شناخت ہوئی ({info.language_probability:.1%} یقین)")
    st.subheader("خام ٹرانسکریپشن")
    st.write(text)

    st.subheader("خوبصورت درست شدہ متن")
    clean_text = text.replace(" ھے ", " ہے ").replace(" اج ", " آج ").replace(" ارہا ", " آ رہا ")
    clean_text = clean_text.replace(" لائی لائی ", " لائ لائ ").replace("ھو", "ہو").replace("ھی", "ہی")
    st.markdown(f"<div dir='rtl' style='font-size:18px; line-height:2'>{clean_text}</div>", unsafe_allow_html=True)

    st.download_button("متن ڈاؤن لوڈ کریں", clean_text, "urdu_transcript.txt")

    os.unlink(audio_path)
else:
    st.info("اوپر فائل ڈالیں – WhatsApp voice, YouTube, lecture سب چلیں گے!")
