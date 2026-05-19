import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def prepare_liar(csv_folder="."):
    train_df = pd.read_csv(f"{csv_folder}/train.csv")
    test_df = pd.read_csv(f"{csv_folder}/test.csv")
    val_df = pd.read_csv(f"{csv_folder}/valid.csv")

    all_data = pd.concat([train_df, test_df, val_df], ignore_index=True)
    all_data['label'] = all_data['label'].apply(lambda x: 1 if x in [0,1,2,5] else 0)
    all_data = all_data.rename(columns={"statement": "text"})

    train, temp = train_test_split(all_data, test_size=0.2, random_state=42, stratify=all_data['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

    train_dataset = Dataset.from_pandas(train[['text','label']])
    val_dataset = Dataset.from_pandas(val[['text','label']])
    test_dataset = Dataset.from_pandas(test[['text','label']])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

if __name__ == "__main__":
    prepare_liar()