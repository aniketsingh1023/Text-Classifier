from fastapi import FastAPI, UploadFile, File
import os
from app.audio_to_text import AudioToText
from app.summarizer import load_and_summarize  # Assuming 'load_and_summarize' is the function to summarize
from app.spam_classifier import SpamClassifier
from app.utils import save_to_json

app = FastAPI()

# Initialize models
audio_model = AudioToText()  # Transcription model
summarizer = load_and_summarize  # Summarizer function
classifier = SpamClassifier()  # Spam classification model

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    input_path = f"data/inputs/{file.filename}"
    
    # Save uploaded audio
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Step 1: Transcribe audio to text
    print("🔄 Transcribing audio...")
    transcribed_text = audio_model.transcribe(input_path)

    # Step 2: Summarize the transcribed text
    print("🧠 Summarizing transcribed text...")
    summarized_text = summarizer(transcribed_text)
    print(f"📝 Summary: {summarized_text}")  # Debug: print the summary

    # Step 3: Classify summarized text as Spam or Not Spam
    print("📦 Classifying text...")
    label, score = classifier.classify(summarized_text)

    # Step 4: Prepare and save the result
    result = {
        "original_text": transcribed_text,
        "summary": summarized_text,
        "classification": label,
        "score": score
    }

    # Save the result in a JSON file
    output_path = f"data/outputs/{file.filename}_result.json"
    print(f"📂 Saving result to: {output_path}")
    save_to_json(result, output_path)

    return result
