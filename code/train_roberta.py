import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""
train_roberta.py
Fine-tune RoBERTa-base on the LIAR dataset for fake news detection.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from data_preprocessing import prepare_liar  # reuse your preprocessing

def compute_metrics(eval_pred):
    """Compute evaluation metrics for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    prec = precision_score(labels, predictions, average='binary')
    rec = recall_score(labels, predictions, average='binary')
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

def tokenize_dataset(dataset, tokenizer, max_length=256):
    """Tokenize text field and return a Hugging Face Dataset."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    tokenized = dataset.map(tokenize_function, batched=True)
    # Keep only necessary columns
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

def train_roberta():
    # 1. Load data
    print("Loading LIAR dataset...")
    liar = prepare_liar()  # returns dict with 'train', 'validation', 'test'
    train_data = liar["train"]
    val_data = liar["validation"]
    test_data = liar["test"]

    # 2. Load tokenizer and model
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # 3. Tokenize datasets
    print("Tokenizing...")
    train_tokenized = tokenize_dataset(train_data, tokenizer)
    val_tokenized = tokenize_dataset(val_data, tokenizer)
    test_tokenized = tokenize_dataset(test_data, tokenizer)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir="./models/roberta_finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=4,                 # 3-4 epochs enough
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,                 # keep only best 2 checkpoints
        fp16=True,                          # faster training on GPU
        dataloader_num_workers=2,
        seed=42,
    )

    # 5. Trainer with early stopping (optional)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    print(f"Test results: {test_results}")

    # 8. Save final model & tokenizer
    trainer.save_model("./models/roberta_finetuned/final")
    tokenizer.save_pretrained("./models/roberta_finetuned/final")
    print("Model saved to ./models/roberta_finetuned/final")

    return trainer

if __name__ == "__main__":
    train_roberta()