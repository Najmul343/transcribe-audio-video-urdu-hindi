import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile

# ============== خوبصورت UI ==============
st.set_page_config(page_title="اردو ٹرانسکرائبر", page_icon="Pakistan", layout="centered")
st.title("Pakistan اردو آڈیو ٹرانسکرائبر")
st.markdown("**WhatsApp وائس، یوٹیوب، لیکچر → فوراً خوبصورت اردو میں**")
st.caption("تیز • مفت • Cloud پر چلتا ہے • 2025")

# ============== ماڈل لوڈ ==============
@st.cache_resource
def load_model():
    with st.spinner("ماڈل لوڈ ہو رہا ہے (صرف پہلی بار)..."):
        return WhisperModel("medium", device="cpu", compute_type="int8")

model = load_model()
st.success("ماڈل تیار ہے!")

# ============== فائل اپ لوڈ ==============
uploaded_file = st.file_uploader(
    "اپنی آڈیو یا ویڈیو فائل ڈالیں",
    type=["mp3", "m4a", "wav", "ogg", "mp4", "webm", "mov"]
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("اردو میں ٹرانسکریپٹ کریں", type="primary"):
        # لازمی: فائل کو ٹیمپ فائل میں سیو کرو
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            audio_path = tmp.name

        try:
            with st.spinner("ٹرانسکریپشن ہو رہی ہے..."):
                segments, info = model.transcribe(
                    audio_path,
                    language="ur",
                    vad_filter=True
                )
                text = " ".join([s.text.strip() for s in segments])

                # عام غلطیاں ٹھیک کرو
                text = text.replace("ھے", "ہے").replace("اج", "آج").replace("ارہا", "آ رہا")
                text = text.replace("لائی لائی", "لائ لائ").replace("ھو", "ہو")

            st.success("کامیاب!")
            st.subheader("خوبصورت اردو متن")
            st.markdown(f"<div dir='rtl' style='font-size:18px; line-height:2;'>{text}</div>", 
                       unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("متن ڈاؤن لوڈ کریں", text, "urdu.txt")
            with col2:
                st.code(f"navigator.clipboard.writeText(`{text}`)")

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

else:
    st.info("اوپر فائل ڈال کر 'ٹرانسکریپٹ کریں' دبائیں")

st.markdown("---")
st.caption("پاکستانیوں کے لیے بنایا گیا • faster-whisper • 2025")
