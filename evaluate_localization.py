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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-sweep",
        default=None,
        help="Comma-separated thresholds, e.g. 0.05,0.1,0.2,0.3,0.4,0.5",
    )
    parser.add_argument(
        "--save-markdown",
        action="store_true",
        help="Save each checkpoint's evaluation result to a markdown file named after the checkpoint.",
    )
    parser.add_argument(
        "--markdown-dir",
        default=None,
        help="Directory to save markdown reports. Defaults to the checkpoint directory.",
    )
    return parser.parse_args()


def parse_threshold_sweep(raw: str | None, default_threshold: float) -> list[float]:
    if raw is None:
        return [default_threshold]
    thresholds: list[float] = []
    for part in raw.split(","):
        value = float(part.strip())
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold values must be in [0, 1]")
        thresholds.append(value)
    if not thresholds:
        raise ValueError("threshold_sweep must contain at least one threshold")
    return thresholds


def format_markdown_table(rows: list[dict[str, float]]) -> str:
    header = "| threshold | loss | accuracy | precision | recall | f1 |"
    divider = "|---:|---:|---:|---:|---:|---:|"
    body = [
        (
            f"| {row['threshold']:.4f} | {row['loss']:.6f} | {row['accuracy']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |"
        )
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def resolve_checkpoint_paths(path: Path) -> list[Path]:
    if path.is_dir():
        checkpoints = sorted(path.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no .pt checkpoints found in directory: {path}")
        return checkpoints
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return [path]


def build_markdown_report(
    checkpoint_path: Path,
    dataset_path: str,
    num_samples: int,
    num_features: int,
    backbone_name: str,
    max_length: int,
    batch_size: int,
    rows: list[dict[str, float]],
    best_threshold: float | None,
    best_f1: float,
) -> str:
    lines = [
        f"# Evaluation Report: `{checkpoint_path.name}`",
        "",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- dataset_path: `{dataset_path}`",
        f"- num_samples: `{num_samples}`",
        f"- num_features: `{num_features}`",
        f"- backbone_name: `{backbone_name}`",
        f"- max_length: `{max_length}`",
        f"- batch_size: `{batch_size}`",
        "",
        format_markdown_table(rows),
    ]
    if best_threshold is not None:
        lines.extend(
            [
                "",
                f"- best_threshold: `{best_threshold:.4f}`",
                f"- best_f1: `{best_f1:.6f}`",
            ]
        )
    return "\n".join(lines) + "\n"


def save_markdown_report(report: str, checkpoint_path: Path, markdown_dir: str | None) -> Path:
    target_dir = Path(markdown_dir) if markdown_dir is not None else checkpoint_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / f"{checkpoint_path.stem}.md"
    report_path.write_text(report)
    return report_path


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_paths = resolve_checkpoint_paths(checkpoint_path)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to evaluate the localization model.") from exc

    thresholds = parse_threshold_sweep(args.threshold_sweep, args.threshold)
    samples_cache = None

    for current_checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(current_checkpoint_path, map_location="cpu")
        backbone_name = args.backbone_name or checkpoint["backbone_name"]
        max_length = args.max_length or checkpoint.get("max_length", 1024)
        pos_weight = float(checkpoint.get("pos_weight", 1.0))

        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        add_localization_special_tokens(tokenizer)

        if samples_cache is None:
            samples_cache = load_localization_jsonl(
                path=args.dataset_path,
                max_samples=args.max_samples,
                show_progress=True,
            )
        samples = samples_cache
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

        print(f"checkpoint_path={current_checkpoint_path}")
        print(f"dataset_path={args.dataset_path}")
        print(f"num_samples={len(samples)}")
        print(f"num_features={len(features)}")
        print(f"backbone_name={backbone_name}")
        print(f"max_length={max_length}")
        print(f"batch_size={args.batch_size}")

        best_threshold = None
        best_f1 = float("-inf")
        rows: list[dict[str, float]] = []
        for threshold in thresholds:
            metrics = evaluate_localization_model(model, dataloader, args.device, threshold=threshold)
            row = {
                "threshold": threshold,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            rows.append(row)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold

        print()
        print(format_markdown_table(rows))

        if len(thresholds) > 1:
            print()
            print(f"best_threshold={best_threshold:.4f}")
            print(f"best_f1={best_f1:.6f}")

        if args.save_markdown:
            report = build_markdown_report(
                checkpoint_path=current_checkpoint_path,
                dataset_path=args.dataset_path,
                num_samples=len(samples),
                num_features=len(features),
                backbone_name=backbone_name,
                max_length=max_length,
                batch_size=args.batch_size,
                rows=rows,
                best_threshold=best_threshold if len(thresholds) > 1 else None,
                best_f1=best_f1,
            )
            report_path = save_markdown_report(
                report=report,
                checkpoint_path=current_checkpoint_path,
                markdown_dir=args.markdown_dir,
            )
            print(f"markdown_report={report_path}")
        print()


if __name__ == "__main__":
    main()
