import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from train.classification.train import CodeT5Classifier, JulietDataset

SAVE_DIR   = "models/classifier"
TEST_PATH  = "data/splits/test.json"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MAX_LENGTH = 512
BATCH_SIZE = 32

with open("data/classification_processed/label_maps.json") as f:
    label_maps = json.load(f)

NUM_CATEGORIES  = len(label_maps["category_to_id"])
NUM_ERROR_TYPES = len(label_maps["error_type_to_id"])


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = CodeT5Classifier("Salesforce/codet5p-220m", NUM_CATEGORIES, NUM_ERROR_TYPES)
    checkpoint = torch.load(f"{SAVE_DIR}/best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    print(f"Loading test set from {TEST_PATH}...")
    dataset = JulietDataset(TEST_PATH, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    cat_preds, cat_labels = [], []
    err_preds, err_labels = [], []

    for i, batch in enumerate(loader):
        with torch.no_grad():
            cat_logits, err_logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
            )
        cat_preds.extend(cat_logits.argmax(1).tolist())
        cat_labels.extend(batch["category_label"].tolist())
        err_preds.extend(err_logits.argmax(1).tolist())
        err_labels.extend(batch["error_type_label"].tolist())

        if (i + 1) % 50 == 0:
            print(f"  {(i + 1) * BATCH_SIZE}/{len(dataset)} samples processed")

    cat_names = [label_maps["id_to_category"][str(i)]   for i in range(NUM_CATEGORIES)]
    err_names = [label_maps["id_to_error_type"][str(i)] for i in range(NUM_ERROR_TYPES)]

    print("\n=== Category F1 per class ===")
    print(classification_report(cat_labels, cat_preds, target_names=cat_names, zero_division=0))

    print("=== Error Type F1 per class ===")
    print(classification_report(err_labels, err_preds, target_names=err_names, zero_division=0))


if __name__ == "__main__":
    main()
