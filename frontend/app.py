# frontend/app.py (Refactored Streamlit UI)
import streamlit as st
import requests
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()
st.set_page_config(page_title="🎧 Voice Assistant", layout="wide")
st.title("🎧 Voice Assistant")
st.markdown("Upload an audio file and get transcription, spam classification, and summarization.")

# Upload UI
uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

if uploaded_file:
    with st.spinner("Processing audio..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post("http://localhost:8000/upload/", files=files)

        if response.status_code == 200:
            result = response.json()

            console.rule("[bold green]🎧 Voice Assistant Result")

            console.print(Panel(f"[bold yellow]🎙️ Transcription:\n[/bold yellow]{result.get('transcription', 'No transcription')}"))
            console.print(Panel(f"[bold magenta]🧪 Classification:[/bold magenta] {result.get('label')} (Score: {result.get('score'):.2f})"))
            console.print(Panel(f"[bold cyan]📝 Summary:\n[/bold cyan]{result.get('summary', 'No summary')}", expand=False))

            st.subheader("🎙️ Transcription")
            st.success(result.get("transcription", "No transcription available."))

            st.subheader("🧪 Spam Classification")
            st.info(f"Label: **{result.get('label')}** | Score: {result.get('score'):.2f}")

            st.subheader("📝 Summary")
            st.warning(result.get("summary", "No summary available."))

        else:
            st.error("Failed to process the audio. Please try again.")
else:
    st.info("Please upload an audio file to get started.")
