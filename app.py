import streamlit as st
from faster_whisper import WhisperModel
import torch
import os
import tempfile
import io

# Page config for RTL Urdu support
st.set_page_config(
    page_title="اردو آڈیو ٹرانسکرائبر",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇵🇰 اردو آڈیو ٹرانسکرائبر")
st.markdown("**WhatsApp وائس، یوٹیوب، لیکچر → فوراً درست اردو متن** | مفت • تیز • Cloud پر چلتا ہے")

# Sidebar for model selection
st.sidebar.header("ماڈل منتخب کریں")
model_size = st.sidebar.selectbox(
    "ماڈل سائز (بڑا = بہتر اردو، چھوٹا = تیز)",
    ["small", "medium", "large-v3"],
    index=1  # Default: medium
)
use_gpu = st.sidebar.checkbox("GPU استعمال کریں (اگر دستیاب ہو)", value=False)  # CPU safe for Cloud

# Load model (cached, one-time)
@st.cache_resource
def load_whisper_model(size):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    with st.spinner(f"{size} ماڈل لوڈ ہو رہا ہے... (صرف پہلی بار)"):
        return WhisperModel(size, device=device, compute_type=compute_type)

model = load_whisper_model(model_size)
st.sidebar.success("ماڈل تیار!")

# File uploader (restrict to audio/video)
uploaded_file = st.file_uploader(
    "اپنی آڈیو/ویڈیو فائل ڈالیں",
    type=["mp3", "m4a", "wav", "ogg", "mp4", "webm", "mov", "flac"],
    help="سب سے بہتر: m4a (WhatsApp) یا mp3 (یوٹیوب)"
)

if uploaded_file is not None:
    # FIXED: Save to temp file path (key fix for av.open() error)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_path = tmp_file.name

    # Preview audio
    st.audio(uploaded_file, format="audio/mpeg")

    # Transcribe button for control
    if st.button("اردو میں ٹرانسکریپٹ کریں", type="primary"):
        with st.spinner("اردو میں ٹرانسکریپشن ہو رہی ہے... (large-v3 = کمال)"):
            try:
                # FIXED: Pass file path (string) – no direct file object!
                segments, info = model.transcribe(
                    audio_path,  # Path, not file object
                    language="ur",  # Urdu
                    vad_filter=True,  # Remove silence
                    beam_size=7  # Better accuracy
                )
                
                full_text = " ".join([seg.text.strip() for seg in segments])
                
                # Simple post-processing for Urdu (fix common Whisper errors)
                full_text = full_text.replace("ھے", "ہے").replace("اج", "آج").replace("ارہا", "آ رہا")
                full_text = full_text.replace("لائی لائی", "لائ لائ").replace("ھو", "ہو")
                full_text = re.sub(r'\s+', ' ', full_text).strip()  # Clean spaces

                st.success(f"کامیاب! زبان: اردو ({info.language_probability:.1%} یقین)")

                # Display results
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("لفظوں کی تعداد", len(full_text.split()))
                with col2:
                    st.subheader("خوبصورت اردو متن")
                    st.markdown(f"<div dir='rtl' style='font-size:18px; line-height:1.8; text-align:right;'>{full_text}</div>", unsafe_allow_html=True)

                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="متن ڈاؤن لوڈ کریں (.txt)",
                        data=full_text,
                        file_name="اردو_ٹرانسکریپشن.txt",
                        mime="text/plain"
                    )
                with col2:
                    st.code(f"navigator.clipboard.writeText(`{full_text}`);", language="javascript")
                    st.caption("اوپر کوڈ کاپی کرکے براؤزر کنسول میں پیسٹ کریں")

            except Exception as e:
                st.error(f"غلطی: {str(e)}. چیک کریں فائل درست ہے (صرف آڈیو/ویڈیو).")

        # Cleanup temp file
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

else:
    st.info("📁 اوپر فائل ڈال کر 'ٹرانسکریپٹ کریں' دبائیں۔ مثال: WhatsApp voice note (.m4a)")
    st.markdown("**ٹپ**: بڑی فائلز (10+ منٹ) کے لیے 'small' ماڈل منتخب کریں – تیز ہو جائے گی!")

# Footer
st.markdown("---")
st.markdown("**پاکستانیوں کے لیے بنایا گیا** • faster-whisper + Streamlit Cloud • 2025")
