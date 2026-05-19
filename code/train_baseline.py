import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

def train_baseline(train_texts, train_labels, val_texts, val_labels):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_val)
    acc = accuracy_score(val_labels, preds)
    f1 = f1_score(val_labels, preds)
    print(f"Baseline Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    joblib.dump(vec, 'models/tfidf_vectorizer.pkl')
    joblib.dump(clf, 'models/logreg_model.pkl')
    return vec, clf

if __name__ == "__main__":
    from data_preprocessing import prepare_liar
    liar = prepare_liar()
    train_texts = liar['train']['text']
    train_labels = liar['train']['label']
    val_texts = liar['validation']['text']
    val_labels = liar['validation']['label']
    train_baseline(train_texts, train_labels, val_texts, val_labels)