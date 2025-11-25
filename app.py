import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="اردو ٹرانسکرائبر", page_icon="Pakistan")

st.title("Pakistan اردو آڈیو → متن")
st.caption("WhatsApp وائس، یوٹیوب، لیکچر – سب کچھ خوبصورت اردو میں")

# ماڈل لوڈ کرو (صرف ایک بار)
@st.cache_resource
def get_model():
    return WhisperModel("medium", device="cpu", compute_type="int8")

model = get_model()
st.success("ماڈل تیار ہے")

# فائل اپ لوڈ
audio_file = st.file_uploader("اپنی آڈیو/ویڈیو ڈالیں", 
                             type=["mp3","m4a","wav","ogg","mp4","webm"])

if audio_file:
    with st.spinner("اردو میں ٹرانسکریپٹ ہو رہا ہے..."):
        segments, _ = model.transcribe(audio_file, language="ur", vad_filter=True)
        text = " ".join([s.text.strip() for s in segments])

    st.balloons()
    st.subheader("تیار شدہ اردو متن")
    st.write(text)
    
    st.download_button("متن ڈاؤن لوڈ کریں", text, "urdu.txt")
else:
    st.info("اوپر فائل ڈال کر شروع کریں")

st.markdown("---")
st.caption("بنایا پاکستانیوں کے لیے • faster-whisper medium • 2025")
