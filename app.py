import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

# ---------------------------------------------------------
# 🟢 Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="اردو ٹرانسکرائبر",
    page_icon="🇵🇰",
    layout="centered"
)

st.title("🇵🇰 پاکستان اردو آڈیو ٹرانسکرائبر")
st.markdown("### **WhatsApp وائس، لیکچر، یوٹیوب → خوبصورت اردو ٹیکسٹ**")
st.caption("2025 • تیز ترین • بغیر خرچے کے • CPU Optimized")

# ---------------------------------------------------------
# 🟢 Load Whisper Model (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

model = load_model()
st.success("ماڈل کامیابی سے لوڈ ہوگیا ✔️")

# ---------------------------------------------------------
# 🟢 File Upload
# ---------------------------------------------------------
file = st.file_uploader(
    "آڈیو یا ویڈیو فائل اپ لوڈ کریں:",
    type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"]
)

# ---------------------------------------------------------
# 🟢 If File Uploaded
# ---------------------------------------------------------
if file:
    st.audio(file, format="audio/mp3")

    if st.button("اردو میں ٹرانسکرائب کریں 🚀"):
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        with st.spinner("مہربانی سے انتظار کریں… ٹرانسکرپشن جاری ہے ⏳"):
            segments, info = model.transcribe(
                temp_path,
                language="ur",
                beam_size=5
            )
            final_text = " ".join([seg.text for seg in segments])

        # Clean up
        os.remove(temp_path)

        st.success("ٹرانسکرپشن مکمل ✔️")
        st.markdown("### 📄 ٹیکسٹ:")

        st.write(final_text)

        st.download_button(
            label="اردو ٹیکسٹ ڈاؤن لوڈ کریں",
            file_name="urdu_transcript.txt",
            data=final_text
        )

        st.balloons()

else:
    st.info("براہ کرم کوئی آڈیو/ویڈیو فائل اپ لوڈ کریں۔")
