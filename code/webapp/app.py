import streamlit as st
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np

@st.cache_resource
def load_models():
    classifier = pipeline("text-classification", model="../models/roberta_progressive")
    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
    return classifier, explainer

st.title("🔎 Fake News Detector")
text = st.text_area("Enter news text:")

if st.button("Analyze"):
    clf, exp = load_models()
    result = clf(text)[0]
    label = "FAKE" if result['label'] == 'LABEL_1' else "REAL"
    confidence = result['score'] if label=="FAKE" else 1-result['score']
    st.write(f"**Prediction:** {label} (confidence: {confidence:.2f})")
    # LIME
    def predict_proba(texts):
        out = clf(texts)
        probs = []
        for o in out:
            if o['label']=='LABEL_1':
                probs.append([1-o['score'], o['score']])
            else:
                probs.append([o['score'], 1-o['score']])
        return np.array(probs)
    explanation = exp.explain_instance(text, predict_proba, num_features=5)
    st.components.v1.html(explanation.as_html(), height=400)