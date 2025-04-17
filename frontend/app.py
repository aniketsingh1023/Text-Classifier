import streamlit as st
import requests
import ffmpeg
from io import BytesIO
from pydub import AudioSegment  # Keep for other potential uses if necessary

# Streamlit page configuration
st.set_page_config(page_title="Multilingual Text Classification", page_icon=":speech_balloon:")

# Define the API URL for your backend
API_URL = "https://your-backend-url-here.com/predict"  # Replace with your backend URL

# Function to convert audio to WAV using ffmpeg
def audio_to_wav(audio_file_path):
    """
    Convert an MP3 file to WAV format using ffmpeg.
    """
    wav_file_path = audio_file_path.replace(".mp3", ".wav")
    ffmpeg.input(audio_file_path).output(wav_file_path).run()
    return wav_file_path

# Function to handle the audio file upload and send to backend
def upload_audio(uploaded_file):
    # Save the uploaded audio file to a temporary location
    audio_file_path = f"temp_{uploaded_file.name}"
    with open(audio_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert to WAV format
    wav_file_path = audio_to_wav(audio_file_path)
    
    # Open the WAV file and prepare it to send
    with open(wav_file_path, 'rb') as f:
        files = {'file': (wav_file_path, f, 'audio/wav')}
        response = requests.post(API_URL, files=files)
    
    return response.json()

# Streamlit UI
st.title("Multilingual Text Classification")
st.write("Upload an audio file (MP3 or WAV) for multilingual text classification.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

# When a file is uploaded
if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")  # Show audio player
    st.write("Processing the audio...")

    # Upload the audio to the backend and get the result
    try:
        result = upload_audio(uploaded_file)
        st.subheader("Prediction Result:")
        st.write(result)
    except requests.exceptions.RequestException as e:
        st.error(f"Error with API request: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
