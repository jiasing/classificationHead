from __future__ import annotations

import json
from pathlib import Path

from data.schema import LocalizationSample
from tqdm.auto import tqdm


def load_localization_jsonl(
    path: str | Path,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> list[LocalizationSample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"localization dataset not found: {dataset_path}")

    samples: list[LocalizationSample] = []
    with dataset_path.open("r") as handle:
        iterator = handle
        if show_progress:
            iterator = tqdm(handle, desc="loading jsonl", unit="samples")
        for line in iterator:
            payload = json.loads(line)
            samples.append(LocalizationSample(**payload))
            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise ValueError(f"no localization samples found in {dataset_path}")

    return samples
