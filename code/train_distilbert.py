import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def tokenize_function(examples, tokenizer, max_len=256):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train_distilbert(train_dataset, val_dataset):
    # Point to the local model folder (relative to project root)
    # Your model is in: TALN_project_fake_news/models/distilbert-base-uncased/
    # The script is in: TALN_project_fake_news/code/
    model_path = "../models/distilbert-base-uncased"
    
    # Verify the folder exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}. Please check the path.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    
    def tokenize(examples):
        return tokenize_function(examples, tokenizer)
    
    train_tokenized = train_dataset.map(tokenize, batched=True)
    val_tokenized = val_dataset.map(tokenize, batched=True)
    train_tokenized = train_tokenized.remove_columns(["text"]).rename_column("label", "labels")
    val_tokenized = val_tokenized.remove_columns(["text"]).rename_column("label", "labels")
    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    training_args = TrainingArguments(
        output_dir="./models/distilbert_finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("./models/distilbert_finetuned")
    tokenizer.save_pretrained("./models/distilbert_finetuned")
    return trainer

if __name__ == "__main__":
    from data_preprocessing import prepare_liar
    liar = prepare_liar()
    train_distilbert(liar['train'], liar['validation'])