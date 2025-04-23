from gtts import gTTS
import uuid
import os

def text_to_speech(summary: str, classification: str, score: float) -> str:
    warning = (
        "This audio was generated using AI. "
        "Please do not blindly trust every output. Consider analyzing further."
    )

    speech_text = (
        f"Summary: {summary}. "
        f"Classification: {classification}. Confidence: {round(score, 2)}. "
        f"{warning}"
    )

    filename = f"data/outputs/{uuid.uuid4()}.mp3"
    tts = gTTS(text=speech_text, lang='en')
    tts.save(filename)

    return filename
