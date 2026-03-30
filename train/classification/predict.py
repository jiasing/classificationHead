# src/predict.py

import torch
import json
from transformers import AutoTokenizer
from src.train import CodeT5Classifier   # reuse the class definition

SAVE_DIR   = "models/classifier"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512

with open("data/processed/label_maps.json") as f:
    label_maps = json.load(f)

NUM_CATEGORIES  = len(label_maps["category_to_id"])
NUM_ERROR_TYPES = len(label_maps["error_type_to_id"])


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = CodeT5Classifier("Salesforce/codet5p-220m", NUM_CATEGORIES, NUM_ERROR_TYPES)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def predict(code: str, model, tokenizer):
    enc = tokenizer(
        code,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        cat_logits, err_logits = model(input_ids, attention_mask)

    cat_id  = cat_logits.argmax(1).item()
    err_id  = err_logits.argmax(1).item()

    category   = label_maps["id_to_category"][str(cat_id)]
    error_type = label_maps["id_to_error_type"][str(err_id)]

    cat_conf = torch.softmax(cat_logits, dim=1).max().item()
    err_conf = torch.softmax(err_logits, dim=1).max().item()

    return {
        "category":        category,
        "category_conf":   round(cat_conf, 3),
        "error_type":      error_type,
        "error_type_conf": round(err_conf, 3),
    }


if __name__ == "__main__":
    model, tokenizer = load_model()

    test_code = """
    void example() {
        char buf[10];
        memcpy(buf, user_input, 100);
    }
    """

    result = predict(test_code, model, tokenizer)
    print(f"Category:   {result['category']}  ({result['category_conf']*100:.1f}%)")
    print(f"Error type: {result['error_type']}  ({result['error_type_conf']*100:.1f}%)")