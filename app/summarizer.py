from transformers import pipeline

def load_and_summarize(text: str):
    print("🚀 Loading summarizer model: google/pegasus-cnn_dailymail")
    summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail", device=-1)

    print("\n🧠 Generating short summary...\n")
    cleaned_text = text.strip().replace("\n", " ")

    # Set chunk size to keep model input under 1024 tokens and limit output length
    chunk = cleaned_text[:1024]  # truncating for large texts

    # Set max_length and min_length for concise summary
    result = summarizer(chunk, max_length=70, min_length=20, do_sample=False)
    print("📦 Raw Result:", result)

    summary_text = result[0]["summary_text"]
    print("🔍 Summary:", summary_text)
    return summary_text


