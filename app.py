import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

st.set_page_config(page_title="اردو ٹرانسکرائبر", page_icon="Pakistan")

st.title("Pakistan اردو آڈیو ٹرانسکرائبر")
st.markdown("**WhatsApp وائس، یوٹیوب، لیکچر → فوراً خوبصورت اردو**")
st.caption("2025 • مفت • کوئی ایرر نہیں")

# ماڈل لوڈ کرو (صرف ایک بار)
@st.cache_resource
def get_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

model = get_model()
st.success("ماڈل تیار!")

# فائل اپ لوڈ
file = st.file_uploader("آڈیو/ویڈیو ڈالیں", type=["mp3","m4a","wav","mp4","webm","ogg"])

if file:
    st.audio(file)
    
    if st.button("اردو میں تبدیل کریں"):
        # فائل کو ٹیمپ میں سیو کرو
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        path = tfile.name
        tfile.close()

        with st.spinner("ٹرانسکریپٹ ہو رہا ہے..."):
            segments, _ = model.transcribe(path, language="ur")
            text = " ".join([s.text for s in segments])

        os.unlink(path)  # ڈیلیٹ کرو

        st.success("تیار ہے!")
        st.write(text)
        st.download_button("ڈاؤن لوڈ", text, "urdu.txt")
else:
    st.info("اوپر فائل ڈال کر بٹن دبائیں")

st.balloons()
