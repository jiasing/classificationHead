# src/split.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "data/classification_processed/3.juliet_labelled.json"

def split_by_cwe(df, train_frac=0.7, val_frac=0.15, seed=42):
    test_frac = 1 - train_frac - val_frac

    # Stratified split for vulnerable samples (have CWE labels)
    vulnerable = df[df['label'] == 1]

    # CWEs with < 10 samples can't be reliably stratified — put them directly in train
    cwe_counts = vulnerable['cwe'].value_counts()
    rare = vulnerable[vulnerable['cwe'].isin(cwe_counts[cwe_counts < 10].index)]
    common = vulnerable[vulnerable['cwe'].isin(cwe_counts[cwe_counts >= 10].index)]

    vuln_train, vuln_temp = train_test_split(common, test_size=1-train_frac, stratify=common['cwe'], random_state=seed)
    vuln_val, vuln_test   = train_test_split(vuln_temp, test_size=test_frac/(val_frac+test_frac), stratify=vuln_temp['cwe'], random_state=seed)
    vuln_train = pd.concat([vuln_train, rare])

    # Clean (label=0) rows have no CWE — split by row
    clean = df[df['label'] == 0].sample(frac=1, random_state=seed)
    n_clean = len(clean)
    clean_train = clean.iloc[:int(n_clean * train_frac)]
    clean_val   = clean.iloc[int(n_clean * train_frac):int(n_clean * (train_frac + val_frac))]
    clean_test  = clean.iloc[int(n_clean * (train_frac + val_frac)):]

    train = pd.concat([vuln_train, clean_train]).sample(frac=1, random_state=seed)
    val   = pd.concat([vuln_val,   clean_val  ]).sample(frac=1, random_state=seed)
    test  = pd.concat([vuln_test,  clean_test ]).sample(frac=1, random_state=seed)

    return train, val, test

if __name__ == "__main__":
    df = pd.read_json(INPUT_FILE, lines=True)
    train, val, test = split_by_cwe(df)

    os.makedirs("data/splits", exist_ok=True)
    train.to_json("data/splits/train.json", orient='records', lines=True)
    val.to_json(  "data/splits/val.json",   orient='records', lines=True)
    test.to_json( "data/splits/test.json",  orient='records', lines=True)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")