# src/train.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CHECKPOINT = "Salesforce/codet5p-220m"
MAX_LENGTH       = 512
BATCH_SIZE       = 16       # reduce to 8 if you get out-of-memory errors
EPOCHS           = 10
LEARNING_RATE    = 2e-5
DROPOUT          = 0.1
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
SAVE_DIR         = "models/classifier"

# Loss weight for each head — 1.0 means equal contribution
# Increase CATEGORY_LOSS_WEIGHT if category accuracy lags behind error type
CATEGORY_LOSS_WEIGHT   = 1.0
ERROR_TYPE_LOSS_WEIGHT = 1.0

# ── Load label maps ───────────────────────────────────────────────────────────

with open("data/classification_processed/label_maps.json") as f:
    label_maps = json.load(f)

NUM_CATEGORIES  = len(label_maps["category_to_id"])    # 4 categories
NUM_ERROR_TYPES = len(label_maps["error_type_to_id"])  # 9 error types

ID_TO_CATEGORY   = label_maps["id_to_category"]
ID_TO_ERROR_TYPE = label_maps["id_to_error_type"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class JulietDataset(Dataset):
    """
    Loads a split JSON file and tokenises functions on the fly.
    Each sample returns:
      - input_ids, attention_mask  → fed into the encoder
      - category_label             → target for the category head
      - error_type_label           → target for the error type head
    """

    def __init__(self, filepath, tokenizer, max_per_class=None):
        df = pd.read_json(filepath, lines=True).reset_index(drop=True)

        # Validate required columns exist
        required = {'function_code', 'category_id', 'error_type_id'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        # Undersample majority classes so no class exceeds max_per_class samples
        if max_per_class is not None:
            df = pd.concat([
                g.sample(min(len(g), max_per_class), random_state=42)
                for _, g in df.groupby("category_id")
            ]).reset_index(drop=True)
            print(f"  After undersampling (max {max_per_class}/class): {len(df)} samples")
            print(f"  Category distribution:\n{df['category_id'].value_counts().sort_index()}\n")

        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enc = self.tokenizer(
            str(row['function_code']),
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "category_label":   torch.tensor(int(row['category_id']),   dtype=torch.long),
            "error_type_label": torch.tensor(int(row['error_type_id']), dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    A small two-layer MLP that sits on top of the encoder output.
    Both heads use the same structure but have independent weights.
    """

    def __init__(self, hidden_size, num_classes, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class CodeT5Classifier(nn.Module):
    """
    CodeT5 encoder with two independent classification heads:
      - category_head:   predicts one of 4 categories
      - error_type_head: predicts one of 9 error types

    Both heads read from the same pooled encoder representation,
    so the encoder is trained jointly to serve both tasks at once.
    """

    def __init__(self, checkpoint, num_categories, num_error_types, dropout=DROPOUT):
        super().__init__()

        # The encoder — this is the pretrained CodeT5 backbone
        # AutoModel loads just the encoder stack, no decoder
        self.encoder = T5EncoderModel.from_pretrained(checkpoint)
        hidden_size = self.encoder.config.hidden_size
        # hidden_size is 768 for codet5p-220m

        # Two independent heads — separate weights, same input
        self.category_head   = ClassificationHead(hidden_size, num_categories,  dropout)
        self.error_type_head = ClassificationHead(hidden_size, num_error_types, dropout)

    def forward(self, input_ids, attention_mask):
        # ── Encode ──────────────────────────────────────────────────────────
        # Pass tokens through the encoder
        # last_hidden_state shape: [batch_size, seq_len, hidden_size]
        #                     e.g. [16, 512, 768]
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        token_vectors = encoder_output.last_hidden_state

        # ── Pool ────────────────────────────────────────────────────────────
        # We need a single vector per sample, not one per token.
        # Masked mean pooling: average only the real tokens, ignore padding.
        #
        # attention_mask is 1 for real tokens, 0 for padding.
        # Expand mask to match hidden dimension so we can multiply elementwise.
        mask = attention_mask.unsqueeze(-1).float()       # [batch, 512, 1]
        sum_vectors  = (token_vectors * mask).sum(dim=1)  # [batch, 768]
        num_tokens   = mask.sum(dim=1).clamp(min=1e-9)   # [batch, 1]  avoid div/0
        pooled = sum_vectors / num_tokens                 # [batch, 768]

        # ── Classify ─────────────────────────────────────────────────────────
        # Both heads receive the same pooled vector independently.
        # Neither head can see the other's output — they are fully separate.
        category_logits   = self.category_head(pooled)    # [batch, 4]
        error_type_logits = self.error_type_head(pooled)  # [batch, 9]

        return category_logits, error_type_logits


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, loader, cat_loss_fn, err_loss_fn):
    """
    Runs a full pass over a dataloader without updating weights.
    Returns average loss, accuracy for each head, and full predictions
    for classification reports.
    """
    model.eval()
    total_loss = 0

    all_cat_preds  = []
    all_cat_labels = []
    all_err_preds  = []
    all_err_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            cat_labels     = batch["category_label"].to(DEVICE)
            err_labels     = batch["error_type_label"].to(DEVICE)

            cat_logits, err_logits = model(input_ids, attention_mask)

            cat_loss = cat_loss_fn(cat_logits, cat_labels)
            err_loss = err_loss_fn(err_logits, err_labels)
            loss = CATEGORY_LOSS_WEIGHT * cat_loss + ERROR_TYPE_LOSS_WEIGHT * err_loss
            total_loss += loss.item()

            all_cat_preds.extend(cat_logits.argmax(1).cpu().numpy())
            all_cat_labels.extend(cat_labels.cpu().numpy())
            all_err_preds.extend(err_logits.argmax(1).cpu().numpy())
            all_err_labels.extend(err_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)

    cat_acc = np.mean(np.array(all_cat_preds) == np.array(all_cat_labels)) * 100
    err_acc = np.mean(np.array(all_err_preds) == np.array(all_err_labels)) * 100

    return avg_loss, cat_acc, err_acc, \
           all_cat_preds, all_cat_labels, \
           all_err_preds, all_err_labels


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    print(f"Device:          {DEVICE}")
    print(f"Categories:      {NUM_CATEGORIES}")
    print(f"Error types:     {NUM_ERROR_TYPES}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Epochs:          {EPOCHS}")
    print()

    # Load tokeniser and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model     = CodeT5Classifier(MODEL_CHECKPOINT, NUM_CATEGORIES, NUM_ERROR_TYPES)
    model.to(DEVICE)

    # Datasets and loaders
    train_dataset = JulietDataset("data/splits/train.json", tokenizer)
    val_dataset   = JulietDataset("data/splits/val.json",   tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2       # parallel data loading — set to 0 on Windows if errors
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # Separate learning rates: encoder gets a smaller LR because it's already
    # pretrained. Heads get the full LR because they start from random weights.
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),       "lr": LEARNING_RATE * 0.1},
        {"params": model.category_head.parameters(), "lr": LEARNING_RATE},
        {"params": model.error_type_head.parameters(),"lr": LEARNING_RATE},
    ])

    # Linear warmup then decay — standard for fine-tuning transformers
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LEARNING_RATE * 0.1, LEARNING_RATE, LEARNING_RATE],
        total_steps=total_steps,
        pct_start=0.1
    )

    # Compute class weights (inverse frequency) to further penalise minority classes
    cat_counts = torch.zeros(NUM_CATEGORIES)
    err_counts = torch.zeros(NUM_ERROR_TYPES)
    for cid, count in train_dataset.df["category_id"].value_counts().items():
        cat_counts[int(cid)] = count
    for eid, count in train_dataset.df["error_type_id"].value_counts().items():
        err_counts[int(eid)] = count
    cat_weights = (cat_counts.sum() / (NUM_CATEGORIES  * cat_counts.clamp(min=1))).to(DEVICE)
    err_weights = (err_counts.sum() / (NUM_ERROR_TYPES * err_counts.clamp(min=1))).to(DEVICE)

    cat_loss_fn = nn.CrossEntropyLoss(weight=cat_weights)
    err_loss_fn = nn.CrossEntropyLoss(weight=err_weights)
    best_cat_macro_f1 = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        train_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            cat_labels     = batch["category_label"].to(DEVICE)
            err_labels     = batch["error_type_label"].to(DEVICE)

            optimizer.zero_grad()

            cat_logits, err_logits = model(input_ids, attention_mask)

            # Combined loss — weighted sum of both head losses
            cat_loss = cat_loss_fn(cat_logits, cat_labels)
            err_loss = err_loss_fn(err_logits, err_labels)
            loss = CATEGORY_LOSS_WEIGHT * cat_loss + ERROR_TYPE_LOSS_WEIGHT * err_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)} "
                      f"| batch loss: {loss.item():.4f} "
                      f"| cat: {cat_loss.item():.4f} "
                      f"| err: {err_loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────────────────────
        val_loss, cat_acc, err_acc, \
        cat_preds, cat_labels, \
        err_preds, err_labels = evaluate(model, val_loader, cat_loss_fn, err_loss_fn)

        cat_macro_f1 = f1_score(cat_labels, cat_preds, average="macro", zero_division=0)

        print(f"\nEpoch {epoch+1}/{EPOCHS} summary")
        print(f"  train loss : {avg_train_loss:.4f}")
        print(f"  val loss   : {val_loss:.4f}")
        print(f"  category accuracy  : {cat_acc:.1f}%")
        print(f"  error type accuracy: {err_acc:.1f}%")
        print(f"  category macro F1  : {cat_macro_f1:.4f}")

        # Per-class breakdown every 3 epochs so you can spot weak classes
        if (epoch + 1) % 3 == 0:
            cat_names = [ID_TO_CATEGORY[str(i)]   for i in range(NUM_CATEGORIES)]
            err_names = [ID_TO_ERROR_TYPE[str(i)]  for i in range(NUM_ERROR_TYPES)]

            print("\n  Category head report:")
            print(classification_report(
                cat_labels, cat_preds,
                labels=list(range(NUM_CATEGORIES)),
                target_names=cat_names,
                zero_division=0
            ))

            print("  Error type head report:")
            print(classification_report(
                err_labels, err_preds,
                labels=list(range(NUM_ERROR_TYPES)),
                target_names=err_names,
                zero_division=0
            ))

        # Save best model checkpoint based on category macro F1
        if cat_macro_f1 > best_cat_macro_f1:
            best_cat_macro_f1 = cat_macro_f1
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "cat_acc":         cat_acc,
                "err_acc":         err_acc,
                "cat_macro_f1":    cat_macro_f1,
            }, f"{SAVE_DIR}/best_model.pt")
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  → checkpoint saved (category macro F1: {cat_macro_f1:.4f})")

        print()

    print("Training complete.")
    print(f"Best category macro F1: {best_cat_macro_f1:.4f}")
    print(f"Model saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    train()