import sys
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_preprocessing import prepare_liar

def evaluate_model(model_path, test_dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    texts = test_dataset['text']
    labels = test_dataset['label']
    preds = []
    probs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, return_token_type_ids=False)
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0,1].item()
            pred = 1 if prob > 0.5 else 0
        preds.append(pred)
        probs.append(prob)
    cm = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'../paper/figures/confusion_matrix_{model_name}.pdf')
    plt.close()
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'../paper/figures/roc_{model_name}.pdf')
    plt.close()
    print(classification_report(labels, preds))
    return cm, roc_auc

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <model_path> <model_name>")
        sys.exit(1)
    model_path = sys.argv[1]
    model_name = sys.argv[2]
    liar = prepare_liar()
    evaluate_model(model_path, liar['test'], model_name)