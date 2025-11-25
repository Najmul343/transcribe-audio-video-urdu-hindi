import streamlit as st
from faster_whisper import WhisperModel
import os

# ============== خوبصورت UI ==============
st.set_page_config(page_title="اردو ٹرانسکرائبر", page_icon="Pakistan", layout="centered")

st.title("Pakistan اردو آڈیو ٹرانسکرائبر")
st.markdown("### WhatsApp وائس، یوٹیوب، لیکچر → فوراً خوبصورت اردو میں")
st.caption("تیز • مفت • کوئی ایرر نہیں • 2025 ایڈیشن")

# ============== ماڈل لوڈ (صرف ایک بار) ==============
@st.cache_resource(show_spinner="ماڈل لوڈ ہو رہا ہے (صرف پہلی بار)...")
def load_model():
    return WhisperModel("medium", device="cpu", compute_type="int8")

model = load_model()
st.success("ماڈل تیار ہے!")

# ============== فائل اپ لوڈ ==============
audio_file = st.file_uploader(
    "اپنی آڈیو یا ویڈیو فائل ڈالیں",
    type=["mp3", "m4a", "wav", "ogg", "mp4", "webm", "mov"],
    help="WhatsApp voice note, YouTube clip, lecture – سب چلے گا!"
)

if audio_file:
    st.audio(audio_file, format="audio/wav")

    with st.spinner("اردو میں ٹرانسکریپٹ ہو رہا ہے... (صبر کریں، کمال ہو رہا ہے)"):
        # faster-whisper خود فائل کو ہینڈل کر لیتا ہے – کوئی کنورژن نہیں!
        segments, info = model.transcribe(audio_file, language="ur", vad_filter=True, beam_size=5)

        full_text = " ".join([seg.text.strip() for seg in segments])

    # ============== نتیجہ دکھاؤ ==============
    st.success(f"اردو شناخت ہوئی! ({info.language_probability:.1%} یقین)")
    
    st.subheader("مکمل درست شدہ اردو متن")
    st.markdown(f"<div dir='rtl' style='font-size:18px; line-height:2;'>{full_text}</div>", unsafe_allow_html=True)

    # ڈاؤن لوڈ اور کاپی بٹن
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("متن ڈاؤن لوڈ کریں", full_text, "اردو_ٹرانسکریپشن.txt", "text/plain")
    with col2:
        st.code(f"navigator.clipboard.writeText(`{full_text}`)", language="javascript")

    st.balloons()

else:
    st.info("اوپر فائل ڈال کر شروع کریں")
    st.markdown("---")
    st.markdown("**سپورٹ شدہ فارمیٹس**: mp3, m4a, wav, mp4, webm, WhatsApp voice notes, YouTube clips")

st.markdown("---")
st.caption("پاکستانیوں کے لیے بنایا گیا • faster-whisper + Streamlit • 2025")
