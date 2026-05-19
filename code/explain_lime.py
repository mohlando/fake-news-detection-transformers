import sys
import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import pipeline

if len(sys.argv) < 2:
    print("Usage: python explain_lime.py <model_path> [text]")
    sys.exit(1)

model_path = sys.argv[1]
text = sys.argv[2] if len(sys.argv) > 2 else "Vaccines cause autism according to a secret study."

classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=-1)

def predict_proba(texts):
    results = classifier(texts, return_all_scores=True)
    probs = []
    for res in results:
        prob_fake = res[1]['score']
        prob_real = res[0]['score']
        probs.append([prob_real, prob_fake])
    return np.array(probs)

explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
exp = explainer.explain_instance(text, predict_proba, num_features=6, labels=[1])
exp.save_to_file(f'../paper/figures/lime_{model_path.replace("/","_")}.html')
print(f"Saved LIME explanation to ../paper/figures/lime_{model_path.replace('/','_')}.html")