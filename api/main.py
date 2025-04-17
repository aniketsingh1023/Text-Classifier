from fastapi import FastAPI, UploadFile, File
import os
from app.audio_to_text import AudioToText
from app.summarizer import Summarizer
from app.spam_classifier import SpamClassifier
from app.utils import save_to_json

app = FastAPI()
audio_model = AudioToText()
summarizer = Summarizer()
classifier = SpamClassifier()

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    input_path = f"data/inputs/{file.filename}"
    
    # Save uploaded audio
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Step 1: Transcribe
    text = audio_model.transcribe(input_path)

    # Step 2: Summarize
    summary = summarizer.summarize(text)

    # Step 3: Classify
    label, score = classifier.classify(text)

    # Step 4: Save and return
    result = {
        "original_text": text,
        "summary": summary,
        "classification": label,
        "score": score
    }

    save_to_json(result, f"data/outputs/{file.filename}.json")
    return result
