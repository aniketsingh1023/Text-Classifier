import streamlit as st
import requests
import json

# Set the title of the app
st.title("Multilingual Spam Detector")

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "flac"])

if uploaded_file is not None:
    # Display uploaded file name
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Send the uploaded file to the FastAPI backend for processing
    try:
        response = requests.post("http://127.0.0.1:8000/upload/", files={"file": uploaded_file})

        if response.status_code == 200:
            result = response.json()
            st.subheader("Original Text:")
            st.write(result["original_text"])

            st.subheader("Summarized Text:")
            st.write(result["summarized_text"])

            st.subheader("Classification Result:")
            st.write(f"Classification: {result['classification']}")
            st.write(f"Score: {result['score']:.4f}")
        else:
            st.error(f"Error: {response.json()['error']}")
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
