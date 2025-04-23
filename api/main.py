from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional
from app.audio_to_text import AudioToText
from app.summarizer import load_and_summarize
from app.spam_classifier import SpamClassifier
from app.utils import append_to_db
from app.tts import text_to_speech
import os
import uuid

app = FastAPI()

audio_model = AudioToText()
summarizer = load_and_summarize
classifier = SpamClassifier()

@app.post("/analyze/")
async def analyze_input(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    transcribed_text = None
    temp_input_path = None

    if file:
        temp_input_path = f"data/inputs/{uuid.uuid4()}_{file.filename}"
        with open(temp_input_path, "wb") as f:
            f.write(await file.read())
        transcribed_text = audio_model.transcribe(temp_input_path)

        # Delete input audio file after transcription
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

    elif text:
        transcribed_text = text
    else:
        return {"error": "Please provide either an audio file or text input."}

    summarized_text = summarizer(transcribed_text)
    label, score = classifier.classify(summarized_text)

    result = {
        "id": str(uuid.uuid4()),
        "original_text": transcribed_text,
        "summary": summarized_text,
        "classification": label,
        "confidence_score": round(score, 4)
    }

    append_to_db(result)

    # Generate TTS with disclaimer
    audio_path = text_to_speech(summarized_text, label, score)
    filename = os.path.basename(audio_path)

    return {
        "result": result,
        "audio_file": f"/get_audio/{filename}"
    }


@app.get("/get_audio/{filename}")
async def get_audio(filename: str):
    file_path = f"data/outputs/{filename}"
    if not os.path.exists(file_path):
        return {"error": "Audio file not found."}
    
    def file_stream_and_cleanup():
        with open(file_path, "rb") as f:
            data = f.read()
        os.remove(file_path)  # Burn after use
        yield data

    return StreamingResponse(file_stream_and_cleanup(), media_type="audio/mp3")
