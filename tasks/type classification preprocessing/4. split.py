# src/split.py

import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/classification_processed/juliet_labelled.json"

def split_by_cwe(df, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)

    cwes = df['cwe'].dropna().unique()
    rng.shuffle(cwes)

    n = len(cwes)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train_cwes = cwes[:train_end]
    val_cwes   = cwes[train_end:val_end]
    test_cwes  = cwes[val_end:]

    # Clean (label=0) rows have no CWE — split them separately by row
    vulnerable = df[df['label'] == 1]
    clean      = df[df['label'] == 0]

    clean_shuffled = clean.sample(frac=1, random_state=seed)
    n_clean = len(clean_shuffled)
    clean_train = clean_shuffled.iloc[:int(n_clean * train_frac)]
    clean_val   = clean_shuffled.iloc[int(n_clean * train_frac):int(n_clean * (train_frac + val_frac))]
    clean_test  = clean_shuffled.iloc[int(n_clean * (train_frac + val_frac)):]

    train = pd.concat([vulnerable[vulnerable['cwe'].isin(train_cwes)], clean_train])
    val   = pd.concat([vulnerable[vulnerable['cwe'].isin(val_cwes)],   clean_val])
    test  = pd.concat([vulnerable[vulnerable['cwe'].isin(test_cwes)],  clean_test])

    return train.sample(frac=1, random_state=seed), \
           val.sample(frac=1, random_state=seed), \
           test.sample(frac=1, random_state=seed)

if __name__ == "__main__":
    df = pd.read_json(INPUT_FILE, lines=True)
    train, val, test = split_by_cwe(df)

    os.makedirs("data/splits", exist_ok=True)
    train.to_json("data/splits/train.json", orient='records', lines=True)
    val.to_json(  "data/splits/val.json",   orient='records', lines=True)
    test.to_json( "data/splits/test.json",  orient='records', lines=True)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")