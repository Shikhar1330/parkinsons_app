import streamlit as st
from utils import predict_from_audio
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Parkinson’s Voice Analyzer",
    page_icon="🧠",
    layout="centered",
)

# --- Header ---
st.title("🧠 Parkinson’s Voice Analysis App")
st.markdown("Upload a voice recording (.wav or .mp3) to analyze Parkinson’s risk, severity, and stress levels.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your voice file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Display player
    st.audio(uploaded_file, format="audio/wav")

    st.info("🔍 Extracting features and running predictions... please wait.")
    try:
        results = predict_from_audio(audio_path)

        st.success("✅ Analysis Complete!")

        # --- Display Results ---
        st.subheader("🧾 Results Summary")
        st.metric("Disease Probability", f"{results['Disease Probability']:.2f}")
        st.metric("Diagnosis", results["Disease Label"])
        st.metric("Predicted Severity", f"{results['Predicted Severity']:.3f}")
        st.metric("Adjusted Severity", f"{results['Adjusted Severity']:.3f}")
        st.metric("Stress Level", results["Stress Label"])
        st.metric("Stress Score", f"{results['Stress Score']:.3f}")

        # --- Interpretation ---
        st.subheader("💬 Interpretation")
        if results["Disease Label"] == "Parkinson's":
            if results["Stress Label"] == "High Stress":
                st.warning("Parkinson’s detected under **high stress**. Stress may be elevating perceived severity.")
            else:
                st.info("Parkinson’s detected with **low stress influence**.")
        else:
            if results["Stress Label"] == "High Stress":
                st.info("No Parkinson’s detected, but stress indicators are **elevated**.")
            else:
                st.success("Normal voice detected — **no Parkinson’s or stress risk**.")

    except Exception as e:
        st.error(f"⚠️ Error during processing: {str(e)}")

    # Clean up temp file
    os.remove(audio_path)
else:
    st.info("Please upload a .wav or .mp3 file to begin analysis.")
