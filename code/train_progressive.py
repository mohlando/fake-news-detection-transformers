import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""
train_progressive.py
Progressive fine-tuning: first on Kaggle (Fake.csv + True.csv), then on LIAR.
"""

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from data_preprocessing import prepare_liar

def load_kaggle_split(fake_path="Fake.csv", real_path="True.csv"):
    """Load Kaggle dataset from two separate files."""
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    fake_df['label'] = 1
    real_df['label'] = 0
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'] + " " + df['text']
    n = len(df)
    train = df[:int(0.8*n)]
    val = df[int(0.8*n):int(0.9*n)]
    test = df[int(0.9*n):]
    return train, val, test

def tokenize_dataset(dataset, tokenizer, max_len=256):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

def train_progressive(fake_csv="Fake.csv", true_csv="True.csv"):
    # Stage 1: Kaggle
    print("Loading Kaggle split files...")
    kaggle_train, kaggle_val, _ = load_kaggle_split(fake_csv, true_csv)
    k_train = Dataset.from_pandas(kaggle_train[['text','label']])
    k_val = Dataset.from_pandas(kaggle_val[['text','label']])
    
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    k_train_tok = tokenize_dataset(k_train, tokenizer)
    k_val_tok = tokenize_dataset(k_val, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./models/progressive_stage1",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=k_train_tok,
        eval_dataset=k_val_tok,
        compute_metrics=compute_metrics,
    )
    print("Stage 1: Training on Kaggle...")
    trainer.train()
    
    # Stage 2: LIAR
    print("Loading LIAR...")
    liar = prepare_liar()
    liar_train = liar["train"]
    liar_val = liar["validation"]
    liar_train_tok = tokenize_dataset(liar_train, tokenizer)
    liar_val_tok = tokenize_dataset(liar_val, tokenizer)
    
    trainer.train_dataset = liar_train_tok
    trainer.eval_dataset = liar_val_tok
    trainer.args.learning_rate = 1e-5
    trainer.args.num_train_epochs = 2
    trainer.args.output_dir = "./models/progressive_stage2"
    print("Stage 2: Fine-tuning on LIAR...")
    trainer.train()
    
    trainer.save_model("./models/roberta_progressive/final")
    tokenizer.save_pretrained("./models/roberta_progressive/final")
    print("Done. Model saved to ./models/roberta_progressive/final")

if __name__ == "__main__":
    train_progressive(fake_csv="Fake.csv", true_csv="True.csv")