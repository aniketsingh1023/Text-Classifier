# api/main.py (Refactored with better error handling and cleaner output)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from app.audio_to_text import AudioToText
from app.spam_classifier import SpamClassifier
from app.summarizer import load_and_summarize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once during startup
asr = AudioToText()
classifier = SpamClassifier()

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    input_path = f"data/inputs/{file.filename}"

    # Save uploaded audio
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Step 1: Transcribe
    transcription = asr.transcribe(input_path)
    if not transcription:
        return {"error": "Transcription failed. Please try with a different audio file."}

    # Step 2: Classify
    label, score = classifier.classify(transcription)

    # Step 3: Summarize
    summary = load_and_summarize(transcription)

    return {
        "transcription": transcription or "[No transcription returned]",
        "label": label or "Unknown",
        "score": score or 0.0,
        "summary": summary or "[No summary available]"
    }
