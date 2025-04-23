import streamlit as st
import requests

st.title("🧠 Multilingual Spam Classifier")

text_input = st.text_area("Enter your text here (or upload audio below):")

audio_file = st.file_uploader("Or upload an audio file (.mp3/.wav):", type=["mp3", "wav"])

if st.button("Analyze"):
    if text_input:
        response = requests.post("http://127.0.0.1:8000/analyze/", data={"text": text_input})
    elif audio_file:
        response = requests.post(
            "http://127.0.0.1:8000/analyze/",
            files={"file": (audio_file.name, audio_file)}
        )
    else:
        st.warning("Please provide either text or an audio file.")
        st.stop()

    if response.status_code == 200:
        result = response.json()["result"]
        st.json(result)

        # Get audio from the returned path
        audio_url = f"http://127.0.0.1:8000{response.json()['audio_file']}"
        audio_response = requests.get(audio_url)

        if audio_response.status_code == 200:
            st.audio(audio_response.content, format="audio/mp3")
        else:
            st.warning("Audio could not be retrieved.")

        if st.button("Analyze Further"):
            st.info("Launching detailed analysis view... (Coming Soon!)")

    else:
        st.error("Something went wrong!")
