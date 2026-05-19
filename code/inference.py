"""
inference.py
Load the trained model and provide prediction + LIME explanation.
Used by the Streamlit app (app.py) to avoid reloading the model on every interaction.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# Global variables to cache model and explainer
_tokenizer = None
_model = None
_explainer = None

def load_model(model_path="./models/roberta_progressive/final"):
    """Load tokenizer and model once and cache them."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model.eval()
    return _tokenizer, _model

def get_explainer(class_names=["REAL", "FAKE"]):
    """Lazy load LIME explainer."""
    global _explainer
    if _explainer is None:
        _explainer = LimeTextExplainer(class_names=class_names)
    return _explainer

def predict_proba(texts):
    """
    Predict probabilities for a list of texts.
    Returns a numpy array of shape (len(texts), 2) where column 0 = REAL prob, column 1 = FAKE prob.
    """
    tokenizer, model = load_model()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs  # shape (n, 2): [REAL_prob, FAKE_prob]

def predict_single(text):
    """
    Return (predicted_label, confidence) for a single text.
    label: "REAL" or "FAKE"
    confidence: probability of the predicted class.
    """
    probs = predict_proba([text])[0]  # shape (2,)
    pred_class = "FAKE" if probs[1] > 0.5 else "REAL"
    confidence = probs[1] if pred_class == "FAKE" else probs[0]
    return pred_class, confidence

def explain_text(text, num_features=6):
    """
    Generate LIME explanation for a single text.
    Returns an Explanation object (can be saved as HTML).
    """
    explainer = get_explainer()
    # LIME needs a function that returns probabilities for each class
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        labels=[1]  # explain only FAKE class (index 1)
    )
    return exp

if __name__ == "__main__":
    # Quick test
    test_text = "The government is hiding evidence of alien contact."
    label, conf = predict_single(test_text)
    print(f"Text: {test_text}")
    print(f"Prediction: {label} (confidence: {conf:.3f})")
    
    # Show LIME explanation as HTML (saves to file)
    exp = explain_text(test_text)
    exp.save_to_file("test_explanation.html")
    print("Explanation saved to test_explanation.html")