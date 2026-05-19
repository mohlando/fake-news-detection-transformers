import joblib
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import prepare_liar

# Load data
liar = prepare_liar()
test_data = liar['test']
texts = test_data['text']
true_labels = test_data['label']

# Load baseline model
vec = joblib.load('../models/tfidf_vectorizer.pkl')
clf = joblib.load('../models/logreg_model.pkl')

# Predict
X_test = vec.transform(texts)
preds = clf.predict(X_test)

# Find misclassified
mis_idx = [i for i, (t,p) in enumerate(zip(true_labels, preds)) if t != p]
print(f"Total misclassified: {len(mis_idx)} out of {len(true_labels)}")

# Table
error_df = pd.DataFrame({
    "Text": [texts[i][:100] for i in mis_idx[:15]],
    "True Label": ["Fake" if true_labels[i]==1 else "Real" for i in mis_idx[:15]],
    "Predicted": ["Fake" if preds[i]==1 else "Real" for i in mis_idx[:15]]
})
error_df.to_csv("../results/error_analysis_baseline.csv", index=False)
print("Saved error table to ../results/error_analysis_baseline.csv")

# Histogram
lengths = [len(texts[i].split()) for i in mis_idx]
plt.figure()
plt.hist(lengths, bins=20, edgecolor='black')
plt.title("Misclassification by text length - Baseline")
plt.xlabel("Number of words")
plt.ylabel("Misclassified count")
plt.savefig("../paper/figures/error_by_length_baseline.pdf")
plt.close()
print("Saved histogram to ../paper/figures/error_by_length_baseline.pdf")