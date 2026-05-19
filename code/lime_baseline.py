import joblib
from lime.lime_text import LimeTextExplainer

# Load the baseline model (TF‑IDF + Logistic Regression)
vec = joblib.load('../models/tfidf_vectorizer.pkl')
clf = joblib.load('../models/logreg_model.pkl')

# Define prediction function for LIME
def predict_proba(texts):
    X = vec.transform(texts)
    return clf.predict_proba(X)

# Create LIME explainer
explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])

# Example text (you can change it)
text = "Vaccines cause autism according to a secret study that was covered up."

# Generate explanation for the FAKE class (label 1)
exp = explainer.explain_instance(text, predict_proba, num_features=6, labels=[1])

# Save as HTML file
exp.save_to_file("../paper/figures/lime_example_baseline.html")
print("LIME explanation saved to ../paper/figures/lime_example_baseline.html")
print("Open this file in a browser and take a screenshot for your paper.")