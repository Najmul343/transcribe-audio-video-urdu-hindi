import streamlit as st
from faster_whisper import WhisperModel
import torch
import os
import tempfile
import re

st.set_page_config(page_title="اردو ٹرانسکرائبر", page_icon="Pakistan", layout="wide")

st.title("Pakistan اردو آڈیو ٹرانسکرائبر")
st.markdown("**WhatsApp وائس، یوٹیوب، لیکچر → فوراً خوبصورت اردو میں**")

# ماڈل سلیکٹ کرو
model_size = st.sidebar.selectbox("ماڈل سائز", ["small", "medium", "large-v3"], index=1)
@st.cache_resource
def load_model():
    return WhisperModel(model_size, device="cpu", compute_type="int8")

model = load_model()
st.sidebar.success(f"{model_size} ماڈل تیار!")

# فائل اپ لوڈ
uploaded_file = st.file_uploader(
    "آڈیو/ویڈیو فائل ڈالیں",
    type=["mp3", "m4a", "wav", "ogg", "mp4", "webm", "mov", "flac"]
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("اردو میں ٹرانسکریپٹ کریں", type="primary"):
        # فائل کو ٹیمپ فائل میں سیو کرو (یہ لازمی ہے!)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            audio_path = tmp.name

        try:
            with st.spinner("ٹرانسکریپشن ہو رہی ہے..."):
                segments, info = model.transcribe(
                    audio_path,
                    language="ur",
                    vad_filter=True,
                    beam_size=5
                )
                text = " ".join([s.text.strip() for s in segments])

                # عام غلطیاں ٹھیک کرو
                text = (text.replace("ھے", "ہے")
                            .replace("اج", "آج")
                            .replace("ارہا", "آ رہا")
                            .replace("لائی لائی", "لائ لائ")
                            .replace("ھو", "ہو")
                            .replace("ھی", "ہی"))

                st.success(f"اردو ({info.language_probability:.1%} یقین)")
                st.subheader("مکمل اردو متن")
                st.markdown(f"<div dir='rtl' style='font-size:18px; line-height:2;'>{text}</div>", 
                           unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("ڈاؤن لوڈ کریں", text, "urdu.txt")
                with col2:
                    st.code(f"navigator.clipboard.writeText(`{text}`)")

        except Exception as e:
            st.error(f"غلطی: {str(e)}")

        finally:
            # ٹیمپ فائل ڈیلیٹ کرو
            if os.path.exists(audio_path):
                os.unlink(audio_path)

else:
    st.info("اوپر فائل ڈال کر 'ٹرانسکریپٹ کریں' دبائیں")

st.markdown("---")
st.caption("پاکستانیوں کے لیے • faster-whisper • 2025")
