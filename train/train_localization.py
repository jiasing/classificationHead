from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


@dataclass(slots=True)
class TrainConfig:
    backbone_name: str = "Salesforce/codet5-base"
    dataset_path: str = "data/localization_c.jsonl"
    max_samples: int | None = None
    val_ratio: float = 0.2
    seed: int = 42
    negative_to_positive_ratio: int = 3
    pos_weight: float = 3.0
    max_length: int = 1024
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "logs/localization"
    model_dir: str = "models/localization"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the line-level localization model.")
    parser.add_argument("--dataset-path", default="data/localization_c.jsonl")
    parser.add_argument("--backbone-name", default="Salesforce/codet5-base")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-to-positive-ratio", type=int, default=3)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", default="logs/localization")
    parser.add_argument("--model-dir", default="models/localization")
    args = parser.parse_args()

    return TrainConfig(
        backbone_name=args.backbone_name,
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        pos_weight=args.pos_weight,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
    )


def split_features(
    features: list[object],
    val_ratio: float,
    seed: int,
) -> tuple[list[object], list[object]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")
    if len(features) < 2 or val_ratio == 0.0:
        return features, []

    num_val = int(len(features) * val_ratio)
    if num_val <= 0:
        num_val = 1
    if num_val >= len(features):
        num_val = len(features) - 1

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(features), generator=generator).tolist()
    val_indices = set(permutation[:num_val])

    train_features = [feature for idx, feature in enumerate(features) if idx not in val_indices]
    val_features = [feature for idx, feature in enumerate(features) if idx in val_indices]
    return train_features, val_features


def is_positive_feature(feature: object) -> bool:
    labels = getattr(feature, "line_labels")
    return any(label == 1 for label in labels)


def downsample_negative_features(
    features: list[object],
    negative_to_positive_ratio: int,
    seed: int,
) -> list[object]:
    if negative_to_positive_ratio < 1:
        raise ValueError("negative_to_positive_ratio must be >= 1")

    positive_features = [feature for feature in features if is_positive_feature(feature)]
    negative_features = [feature for feature in features if not is_positive_feature(feature)]
    if not positive_features or not negative_features:
        return features

    max_negatives = min(len(negative_features), len(positive_features) * negative_to_positive_ratio)
    generator = torch.Generator().manual_seed(seed)
    negative_indices = torch.randperm(len(negative_features), generator=generator).tolist()[:max_negatives]
    sampled_negatives = [negative_features[idx] for idx in negative_indices]

    combined = positive_features + sampled_negatives
    shuffled_indices = torch.randperm(len(combined), generator=generator).tolist()
    return [combined[idx] for idx in shuffled_indices]


def describe_feature_split(name: str, features: list[object]) -> str:
    positives = sum(1 for feature in features if is_positive_feature(feature))
    negatives = len(features) - positives
    return f"{name}: total={len(features)} positive={positives} negative={negatives}"


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader | None, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to train the localization model.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    add_localization_special_tokens(tokenizer)

    samples = load_localization_jsonl(
        path=config.dataset_path,
        max_samples=config.max_samples,
    )
    features = prepare_localization_features(
        samples=samples,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    if not features:
        raise RuntimeError(
            f"No localization features were produced from {config.dataset_path}. "
            "Increase max_length or inspect the dataset formatting."
        )
    train_features, val_features = split_features(
        features=features,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    train_features = downsample_negative_features(
        features=train_features,
        negative_to_positive_ratio=config.negative_to_positive_ratio,
        seed=config.seed,
    )
    val_features = downsample_negative_features(
        features=val_features,
        negative_to_positive_ratio=config.negative_to_positive_ratio,
        seed=config.seed + 1,
    )
    print(describe_feature_split("train", train_features))
    print(describe_feature_split("val", val_features))

    train_dataset = LocalizationDataset(train_features)
    collator = LocalizationCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_dataloader: DataLoader | None = None
    if val_features:
        val_dataset = LocalizationDataset(val_features)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    return train_dataloader, val_dataloader, tokenizer


def evaluate_localization_model(
    model: torch.nn.Module,
    dataloader: DataLoader | None,
    device: str,
) -> dict[str, float]:
    if dataloader is None:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}

    model.eval()
    total_loss = 0.0
    batch_count = 0
    predictions_list: list[Tensor] = []
    gold_list: list[Tensor] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="val", leave=False):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                line_token_positions=batch["line_token_positions"].to(device),
                line_labels=batch["line_labels"].to(device),
                line_mask=batch["line_mask"].to(device),
            )
            total_loss += outputs["loss"].item()
            batch_count += 1
            valid_mask = batch["line_mask"] > 0
            batch_predictions = (torch.sigmoid(outputs["line_logits"].detach().cpu()) >= 0.5).to(torch.long)
            batch_gold = batch["line_labels"].detach().cpu().to(torch.long)
            predictions_list.append(batch_predictions[valid_mask])
            gold_list.append(batch_gold[valid_mask])

    metrics = {"loss": total_loss / max(batch_count, 1)}
    if predictions_list:
        pred_valid = torch.cat(predictions_list, dim=0)
        gold_valid = torch.cat(gold_list, dim=0)
        correct = (pred_valid == gold_valid).sum().item()
        total = gold_valid.numel()
        true_positive = ((pred_valid == 1) & (gold_valid == 1)).sum().item()
        false_positive = ((pred_valid == 1) & (gold_valid == 0)).sum().item()
        false_negative = ((pred_valid == 0) & (gold_valid == 1)).sum().item()
        accuracy = correct / total if total else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 0.0 if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)
        metrics.update({"accuracy": accuracy, "f1": f1})
    else:
        metrics.update({"accuracy": 0.0, "f1": 0.0})
    model.train()
    return metrics


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: object,
    config: TrainConfig,
    filename: str,
    extra_state: dict[str, float | int | str] | None = None,
) -> None:
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)

    state = {
        "backbone_name": config.backbone_name,
        "model_state_dict": model.state_dict(),
        "max_length": config.max_length,
        "line_token": "<LINE>",
        "pos_weight": config.pos_weight,
    }
    if extra_state is not None:
        state.update(extra_state)
    torch.save(state, model_dir / filename)


def train_localization_model(config: TrainConfig) -> None:
    train_dataloader, val_dataloader, tokenizer = build_dataloaders(config)
    model = build_localization_model(
        config.backbone_name,
        tokenizer,
        pos_weight=config.pos_weight,
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(log_dir=config.log_dir)
    global_step = 0
    best_val_accuracy = float("-inf")
    best_val_f1 = float("-inf")

    model.train()
    try:
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0

            train_progress = tqdm(
                train_dataloader,
                desc=f"train epoch {epoch + 1}/{config.num_epochs}",
                leave=True,
            )
            for batch in train_progress:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(config.device),
                    attention_mask=batch["attention_mask"].to(config.device),
                    line_token_positions=batch["line_token_positions"].to(config.device),
                    line_labels=batch["line_labels"].to(config.device),
                    line_mask=batch["line_mask"].to(config.device),
                )
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                epoch_loss += loss_value
                writer.add_scalar("train/loss_step", loss_value, global_step)
                global_step += 1
                train_progress.set_postfix(loss=f"{loss_value:.4f}")

            train_loss = epoch_loss / max(len(train_dataloader), 1)
            writer.add_scalar("train/loss_epoch", train_loss, epoch)

            val_metrics = evaluate_localization_model(model, val_dataloader, config.device)
            writer.add_scalar("val/loss_epoch", val_metrics["loss"], epoch)
            writer.add_scalar("val/accuracy_epoch", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/f1_epoch", val_metrics["f1"], epoch)
            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_accuracy={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            if val_dataloader is not None and val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                save_checkpoint(
                    model,
                    tokenizer,
                    config,
                    filename="best_accuracy.pt",
                    extra_state={
                        "epoch": epoch + 1,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1"],
                        "val_loss": val_metrics["loss"],
                    },
                )

            if val_dataloader is not None and val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                save_checkpoint(
                    model,
                    tokenizer,
                    config,
                    filename="best_f1.pt",
                    extra_state={
                        "epoch": epoch + 1,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1"],
                        "val_loss": val_metrics["loss"],
                    },
                )

        save_localization_artifacts(
            model,
            tokenizer,
            config,
            extra_state={
                "best_val_accuracy": best_val_accuracy if val_dataloader is not None else 0.0,
                "best_val_f1": best_val_f1 if val_dataloader is not None else 0.0,
            },
        )
    finally:
        writer.close()


def save_localization_artifacts(
    model: torch.nn.Module,
    tokenizer: object,
    config: TrainConfig,
    extra_state: dict[str, float | int | str] | None = None,
) -> None:
    save_checkpoint(
        model,
        tokenizer,
        config,
        filename="checkpoint.pt",
        extra_state=extra_state,
    )
    print(f"saved model artifacts to {config.model_dir}")


def main() -> None:
    config = parse_args()
    print(f"dataset_path={config.dataset_path}")
    if config.max_samples is not None:
        print(f"max_samples={config.max_samples}")
    print(f"max_length={config.max_length}")
    print(f"negative_to_positive_ratio=1:{config.negative_to_positive_ratio}")
    print(f"pos_weight={config.pos_weight}")
    train_localization_model(config)


if __name__ == "__main__":
    main()
