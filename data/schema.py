from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LocalizationSample:
    sample_id: str
    language: str
    code: str
    lines: list[str]
    line_labels: list[int]

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("lines must not be empty")
        if len(self.lines) != len(self.line_labels):
            raise ValueError("lines and line_labels must have the same length")
        if any(label not in (0, 1) for label in self.line_labels):
            raise ValueError("line_labels must contain only binary values")

        normalized_code = self.code.rstrip("\n")
        joined_lines = "\n".join(self.lines)
        if normalized_code != joined_lines:
            raise ValueError("code must match lines joined by newline characters")

    @property
    def vulnerable_line_indices(self) -> list[int]:
        return [idx for idx, label in enumerate(self.line_labels) if label == 1]

    @property
    def has_vulnerability(self) -> bool:
        return any(self.line_labels)
