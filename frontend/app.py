import streamlit as st
import requests

# Set the FastAPI server URL (assuming it's running locally)
API_URL = "http://127.0.0.1:8000/upload/"

def upload_audio(file):
    # Prepare the file for upload
    files = {'file': file}
    response = requests.post(API_URL, files=files)

    # Handle the response
    if response.status_code == 200:
        st.write("Response received successfully!")
        return response.json()  # Return the result from FastAPI
    else:
        st.error(f"Error: {response.status_code}")
        return None

# Streamlit UI
st.title("Audio Transcription, Summarization, and Spam Classification")

# File uploader widget
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Display the audio file name
    st.audio(uploaded_file, format="audio/wav")

    # Display a loading spinner while processing
    with st.spinner("Processing your audio..."):
        result = upload_audio(uploaded_file)

    if result:
        # Debug: Show the full response
        st.write("Full Response from Backend:")
        st.json(result)

        # Display the results from the FastAPI backend
        st.subheader("Original Transcription")
        st.write(result["original_text"])

        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Classification: ")
        st.write(f"Label: {result['classification']}")
        st.write(f"Confidence Score: {result['score']}")
