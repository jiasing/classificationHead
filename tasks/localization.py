from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from data.schema import LocalizationSample

LINE_TOKEN = "<LINE>"
IGNORE_LABEL = -100.0


def build_localization_text(sample: LocalizationSample, line_token: str = LINE_TOKEN) -> str:
    return "\n".join(f"{line_token} {line}" for line in sample.lines)


@dataclass(slots=True)
class LocalizationFeatures:
    sample_id: str
    input_ids: list[int]
    attention_mask: list[int]
    line_token_positions: list[int]
    line_labels: list[int]


class LocalizationDataset(torch.utils.data.Dataset[LocalizationFeatures]):
    def __init__(self, features: list[LocalizationFeatures]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> LocalizationFeatures:
        return self.features[index]


class LocalizationCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[LocalizationFeatures]) -> dict[str, Tensor | list[str]]:
        batch_size = len(batch)
        max_seq_len = max(len(item.input_ids) for item in batch)
        max_lines = max(len(item.line_labels) for item in batch)

        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        line_token_positions = torch.zeros((batch_size, max_lines), dtype=torch.long)
        line_labels = torch.full((batch_size, max_lines), IGNORE_LABEL, dtype=torch.float)
        line_mask = torch.zeros((batch_size, max_lines), dtype=torch.float)

        sample_ids: list[str] = []
        for row, item in enumerate(batch):
            seq_len = len(item.input_ids)
            num_lines = len(item.line_labels)

            input_ids[row, :seq_len] = torch.tensor(item.input_ids, dtype=torch.long)
            attention_mask[row, :seq_len] = torch.tensor(item.attention_mask, dtype=torch.long)
            line_token_positions[row, :num_lines] = torch.tensor(
                item.line_token_positions,
                dtype=torch.long,
            )
            line_labels[row, :num_lines] = torch.tensor(item.line_labels, dtype=torch.float)
            line_mask[row, :num_lines] = 1.0
            sample_ids.append(item.sample_id)

        return {
            "sample_ids": sample_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "line_token_positions": line_token_positions,
            "line_labels": line_labels,
            "line_mask": line_mask,
        }


def prepare_localization_features(
    samples: list[LocalizationSample],
    tokenizer: Any,
    max_length: int = 512,
    line_token: str = LINE_TOKEN,
) -> list[LocalizationFeatures]:
    features: list[LocalizationFeatures] = []

    line_token_id = tokenizer.convert_tokens_to_ids(line_token)
    if line_token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"{line_token!r} is not registered in tokenizer vocabulary. "
            "Call add_localization_special_tokens() first."
        )

    for sample in samples:
        text = build_localization_text(sample, line_token=line_token)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            add_special_tokens=True,
        )

        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        line_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == line_token_id]

        if not line_positions:
            continue

        kept_labels = sample.line_labels[: len(line_positions)]
        kept_positions = line_positions[: len(kept_labels)]
        if not kept_labels:
            continue

        features.append(
            LocalizationFeatures(
                sample_id=sample.sample_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                line_token_positions=kept_positions,
                line_labels=kept_labels,
            )
        )

    return features


def add_localization_special_tokens(tokenizer: Any) -> int:
    return tokenizer.add_special_tokens({"additional_special_tokens": [LINE_TOKEN]})


class CodeT5LineLocalizationModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.pos_weight = pos_weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        line_token_positions: Tensor,
        line_labels: Tensor | None = None,
        line_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        hidden_size = hidden_states.size(-1)
        gather_index = line_token_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        line_embeddings = torch.gather(hidden_states, dim=1, index=gather_index)
        logits = self.classifier(self.dropout(line_embeddings)).squeeze(-1)

        result: dict[str, Tensor] = {"line_logits": logits}

        if line_labels is not None and line_mask is not None:
            pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
            losses = loss_fn(logits, line_labels)
            masked_loss = (losses * line_mask).sum() / line_mask.sum().clamp_min(1.0)
            result["loss"] = masked_loss

        return result


def build_localization_model(
    backbone_name: str,
    tokenizer: Any,
    pos_weight: float = 1.0,
) -> CodeT5LineLocalizationModel:
    try:
        from transformers import T5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required to build the localization model."
        ) from exc

    encoder = T5EncoderModel.from_pretrained(backbone_name)
    encoder.resize_token_embeddings(len(tokenizer))
    hidden_size = encoder.config.hidden_size
    return CodeT5LineLocalizationModel(
        encoder=encoder,
        hidden_size=hidden_size,
        pos_weight=pos_weight,
    )


def debug_print_tokenization(backbone_name: str = "Salesforce/codet5-base", sample_index: int = 0) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to inspect localization tokenization.") from exc

    from data import TOY_LOCALIZATION_DATASET

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    add_localization_special_tokens(tokenizer)

    sample = TOY_LOCALIZATION_DATASET[sample_index]
    formatted_text = build_localization_text(sample)
    features = prepare_localization_features([sample], tokenizer=tokenizer, max_length=512)
    if not features:
        raise RuntimeError("No features were produced for the selected sample.")

    feature = features[0]
    decoded_tokens = tokenizer.convert_ids_to_tokens(feature.input_ids)

    print("sample_id:", sample.sample_id)
    print("language:", sample.language)
    print()
    print("[original code]")
    print(sample.code)
    print()
    print("[formatted text]")
    print(formatted_text)
    print()
    print("[line labels]")
    print(sample.line_labels)
    print()
    print("[line token positions]")
    print(feature.line_token_positions)
    print()
    print("[tokens]")
    for index, (token_id, token) in enumerate(zip(feature.input_ids, decoded_tokens)):
        marker = "" if index in feature.line_token_positions else ""
        print(f"{index:03d}: id={token_id:5d} token={token}{marker}")

if __name__ == "__main__":
    debug_print_tokenization()
