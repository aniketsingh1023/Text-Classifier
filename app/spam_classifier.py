# app/spam_classifier.py

from transformers import BertTokenizer, BertForSequenceClassification
import torch

class SpamClassifier:
    def __init__(self):
        # Load the pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
        self.model.eval()  # Put the model in evaluation mode

    def classify(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        # Assuming 0 is 'not spam' and 1 is 'spam'
        label = "spam" if predicted_class == 1 else "not spam"
        score = torch.softmax(logits, dim=-1).max().item()

        return label, score
