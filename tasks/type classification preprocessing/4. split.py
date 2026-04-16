# src/split.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "data/classification_processed/3.juliet_labelled.json"

def split_by_cwe(df, train_frac=0.7, val_frac=0.15, seed=42):
    test_frac = 1 - train_frac - val_frac

    # CWEs with < 10 samples can't be reliably stratified — put them directly in train
    cwe_counts = df['cwe'].value_counts()
    rare   = df[df['cwe'].isin(cwe_counts[cwe_counts < 10].index)]
    common = df[df['cwe'].isin(cwe_counts[cwe_counts >= 10].index)]

    train, temp = train_test_split(common, test_size=1-train_frac, stratify=common['cwe'], random_state=seed)
    val, test   = train_test_split(temp, test_size=test_frac/(val_frac+test_frac), stratify=temp['cwe'], random_state=seed)
    train = pd.concat([train, rare]).sample(frac=1, random_state=seed)

    return train, val, test

if __name__ == "__main__":
    df = pd.read_json(INPUT_FILE, lines=True)
    train, val, test = split_by_cwe(df)

    os.makedirs("data/splits", exist_ok=True)
    train.to_json("data/splits/train.json", orient='records', lines=True)
    val.to_json(  "data/splits/val.json",   orient='records', lines=True)
    test.to_json( "data/splits/test.json",  orient='records', lines=True)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")