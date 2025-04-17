import streamlit as st
import requests
import whisper
import torch
from transformers import pipeline
import tempfile
import os
from pydub import AudioSegment
from io import BytesIO

# Load the Whisper model
whisper_model = whisper.load_model("base")

# Load the HuggingFace text classification model (spam classification)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the possible labels for spam classification
labels = ["spam", "not spam"]

def transcribe_audio(audio_file):
    """
    Transcribe the given audio file using Whisper.
    """
    # Load audio file
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # Make a prediction
    result = whisper_model.transcribe(audio)

    # Return the transcribed text
    return result["text"]

def classify_text(text):
    """
    Classify the given text into 'spam' or 'not spam' using HuggingFace's zero-shot classification model.
    """
    # Get predictions for the given text
    result = classifier(text, candidate_labels=labels)
    
    # Return the classification result
    return result['labels'][0], result['scores'][0]

def save_audio_file(uploaded_file):
    """
    Save the uploaded file in a temporary location.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio_file:
        tmp_audio_file.write(uploaded_file.getbuffer())
        return tmp_audio_file.name

def audio_to_wav(audio_file_path):
    """
    Convert an MP3 file to WAV format using pydub.
    """
    audio = AudioSegment.from_mp3(audio_file_path)
    wav_file_path = audio_file_path.replace(".mp3", ".wav")
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def main():
    """
    Main function to create the Streamlit app UI and handle file upload and processing.
    """
    st.title("Multilingual Audio Classification (Spam or Not Spam)")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav"])
    
    if uploaded_file:
        # Save the uploaded audio file
        audio_file_path = save_audio_file(uploaded_file)

        # If the file is in MP3 format, convert it to WAV for compatibility with Whisper
        if audio_file_path.endswith('.mp3'):
            audio_file_path = audio_to_wav(audio_file_path)

        # Transcribe the audio to text
        st.subheader("Transcription:")
        transcribed_text = transcribe_audio(audio_file_path)
        st.write(transcribed_text)

        # Classify the transcribed text
        st.subheader("Classification:")
        classification, confidence = classify_text(transcribed_text)
        st.write(f"Classification: {classification} (Confidence: {confidence:.2f})")

        # Clean up temporary files
        os.remove(audio_file_path)

if __name__ == "__main__":
    main()
