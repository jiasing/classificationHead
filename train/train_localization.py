from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import TOY_LOCALIZATION_DATASET
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
    max_length: int = 512
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "logs/localization"
    model_dir: str = "models/localization"


def build_dataloader(config: TrainConfig) -> tuple[DataLoader, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to train the localization model.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    add_localization_special_tokens(tokenizer)

    features = prepare_localization_features(
        samples=TOY_LOCALIZATION_DATASET,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    dataset = LocalizationDataset(features)
    collator = LocalizationCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    return dataloader, tokenizer


def train_localization_model(config: TrainConfig) -> None:
    dataloader, tokenizer = build_dataloader(config)
    model = build_localization_model(config.backbone_name, tokenizer).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(log_dir=config.log_dir)
    global_step = 0

    model.train()
    try:
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0

            for batch in dataloader:
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

            avg_loss = epoch_loss / max(len(dataloader), 1)
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f}")

        save_localization_artifacts(model, tokenizer, config)
    finally:
        writer.close()


def save_localization_artifacts(model: torch.nn.Module, tokenizer: object, config: TrainConfig) -> None:
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(model_dir)
    torch.save(
        {
            "backbone_name": config.backbone_name,
            "model_state_dict": model.state_dict(),
            "max_length": config.max_length,
            "line_token": "<LINE>",
        },
        model_dir / "checkpoint.pt",
    )
    print(f"saved model artifacts to {model_dir}")


def main() -> None:
    config = TrainConfig()
    train_localization_model(config)


if __name__ == "__main__":
    main()
