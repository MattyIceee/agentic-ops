"""Loader for labeled Renovate PR eval examples.

Examples are JSON files stored in evals/data/. Each file contains one example.
Run ``load_examples()`` to read them all into ``EvalExample`` objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class StoredEvidence(BaseModel):
    """A pre-fetched evidence item stored alongside a labeled example."""

    source: str
    url: str | None = None
    text: str


class EvalExample(BaseModel):
    """A single labeled Renovate PR used for eval.

    Fields
    ------
    pr_ref:              Human-readable PR reference, e.g. ``"owner/repo#42"``.
    dependency:          Package name, e.g. ``"lodash"``.
    old_version:         Version before the bump, e.g. ``"4.17.20"``.
    new_version:         Version after the bump, e.g. ``"4.17.21"``.
    has_breaking_change: Ground-truth label — True if the bump is known breaking.
    notes:               Optional commentary on why this label was assigned.
    evidence:            Pre-fetched evidence items used in place of live fetches.
    """

    pr_ref: str
    dependency: str
    old_version: str
    new_version: str
    has_breaking_change: bool
    notes: str = ""
    evidence: list[StoredEvidence] = []


def load_examples(path: str | Path = "evals/data") -> list[EvalExample]:
    """Read all JSON files in *path* and return a list of ``EvalExample`` objects.

    Returns an empty list if the directory is empty or does not exist — no crash.
    """
    data_dir = Path(path)
    if not data_dir.is_dir():
        return []

    examples: list[EvalExample] = []
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            raw: Any = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                examples.extend(EvalExample.model_validate(item) for item in raw)
            else:
                examples.append(EvalExample.model_validate(raw))
        except Exception as exc:
            print(f"[evals/loader] Skipping {json_file.name}: {exc}")

    return examples
