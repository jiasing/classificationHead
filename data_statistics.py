from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from transformers import logging as transformers_logging


LINE_TOKEN = "<LINE>"
BATCH_SIZE = 256


def build_localization_text(lines: list[str]) -> str:
    return "\n".join(f"{LINE_TOKEN} {line}" for line in lines)


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = position - lower
    return lower_value + (upper_value - lower_value) * weight


def summarize(values: list[int]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "min": float(sorted_values[0]),
        "mean": statistics.fmean(sorted_values),
        "median": percentile(sorted_values, 0.5),
        "p90": percentile(sorted_values, 0.9),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": float(sorted_values[-1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset statistics for localization JSONL.")
    parser.add_argument("--dataset-path", default="data/localization_c.jsonl")
    parser.add_argument("--backbone-name", default="Salesforce/codet5-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_tokenizer(backbone_name: str) -> Any:
    from transformers import AutoTokenizer

    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [LINE_TOKEN]})
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    tokenizer = load_tokenizer(args.backbone_name)

    line_counts: list[int] = []
    char_counts: list[int] = []
    token_counts: list[int] = []
    vulnerable_line_counts: list[int] = []

    num_samples = 0
    num_positive_samples = 0
    num_clean_samples = 0
    total_positive_lines = 0
    truncated_samples = 0
    formatted_batch: list[str] = []

    def flush_batch() -> None:
        nonlocal truncated_samples
        if not formatted_batch:
            return
        batch_token_ids = tokenizer(
            formatted_batch,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        for input_ids in batch_token_ids:
            token_count = len(input_ids)
            token_counts.append(token_count)
            if token_count > args.max_length:
                truncated_samples += 1
        formatted_batch.clear()

    with dataset_path.open("r") as handle:
        for raw_line in handle:
            payload = json.loads(raw_line)
            lines: list[str] = payload["lines"]
            labels: list[int] = payload["line_labels"]

            num_samples += 1
            line_count = len(lines)
            positive_lines = sum(labels)
            char_count = len(payload["code"])

            line_counts.append(line_count)
            char_counts.append(char_count)
            vulnerable_line_counts.append(positive_lines)
            total_positive_lines += positive_lines
            formatted_batch.append(build_localization_text(lines))

            if positive_lines > 0:
                num_positive_samples += 1
            else:
                num_clean_samples += 1

            if len(formatted_batch) >= BATCH_SIZE:
                flush_batch()

            if args.max_samples is not None and num_samples >= args.max_samples:
                break

    flush_batch()

    print(f"dataset_path={dataset_path}")
    print(f"backbone_name={args.backbone_name}")
    print(f"samples={num_samples}")
    print(f"positive_samples={num_positive_samples}")
    print(f"clean_samples={num_clean_samples}")
    print(f"positive_sample_ratio={num_positive_samples / num_samples:.6f}")
    print(f"total_positive_lines={total_positive_lines}")
    print(f"average_positive_lines_per_sample={total_positive_lines / num_samples:.6f}")
    print(f"truncated_samples_at_max_length_{args.max_length}={truncated_samples}")
    print(f"truncated_ratio_at_max_length_{args.max_length}={truncated_samples / num_samples:.6f}")

    for name, values in (
        ("line_count", line_counts),
        ("char_count", char_counts),
        ("token_count", token_counts),
        ("positive_line_count", vulnerable_line_counts),
    ):
        stats = summarize(values)
        print(f"[{name}]")
        for key in ("min", "mean", "median", "p90", "p95", "p99", "max"):
            print(f"{key}={stats[key]:.3f}")


if __name__ == "__main__":
    main()
