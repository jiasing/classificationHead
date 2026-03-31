from __future__ import annotations

import json
from pathlib import Path

from data.schema import LocalizationSample


def load_localization_jsonl(path: str | Path, max_samples: int | None = None) -> list[LocalizationSample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"localization dataset not found: {dataset_path}")

    samples: list[LocalizationSample] = []
    with dataset_path.open("r") as handle:
        for line in handle:
            payload = json.loads(line)
            samples.append(LocalizationSample(**payload))
            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise ValueError(f"no localization samples found in {dataset_path}")

    return samples
