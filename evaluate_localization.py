from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_localization_jsonl
from tasks.localization import (
    LocalizationCollator,
    LocalizationDataset,
    add_localization_special_tokens,
    build_localization_model,
    prepare_localization_features,
)
from train.train_localization import evaluate_localization_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a localization checkpoint on a JSONL dataset.")
    parser.add_argument("--checkpoint-path", default="models/localization/best_f1.pt")
    parser.add_argument("--dataset-path", default="data/vul_localization_func_c_balanced.jsonl")
    parser.add_argument("--backbone-name", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone_name = args.backbone_name or checkpoint["backbone_name"]
    max_length = args.max_length or checkpoint.get("max_length", 1024)
    pos_weight = float(checkpoint.get("pos_weight", 1.0))

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to evaluate the localization model.") from exc

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    add_localization_special_tokens(tokenizer)

    samples = load_localization_jsonl(
        path=args.dataset_path,
        max_samples=args.max_samples,
        show_progress=True,
    )
    features = prepare_localization_features(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length,
        show_progress=True,
    )
    if not features:
        raise RuntimeError(
            f"No localization features were produced from {args.dataset_path}. "
            "Increase max_length or inspect the dataset formatting."
        )

    dataset = LocalizationDataset(features)
    collator = LocalizationCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = build_localization_model(
        backbone_name=backbone_name,
        tokenizer=tokenizer,
        pos_weight=pos_weight,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_localization_model(model, dataloader, args.device)

    print(f"checkpoint_path={checkpoint_path}")
    print(f"dataset_path={args.dataset_path}")
    print(f"num_samples={len(samples)}")
    print(f"num_features={len(features)}")
    print(f"backbone_name={backbone_name}")
    print(f"max_length={max_length}")
    print(f"batch_size={args.batch_size}")
    print(f"loss={metrics['loss']:.6f}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"precision={metrics['precision']:.6f}")
    print(f"recall={metrics['recall']:.6f}")
    print(f"f1={metrics['f1']:.6f}")


if __name__ == "__main__":
    main()
