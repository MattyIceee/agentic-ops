"""Eval runner: precision/recall/F1 for breaking-change extraction.

Runs Graph A's ``extract_breaking_changes`` node in isolation against each
labeled example's stored evidence, then compares the predicted label
(any findings → breaking) against the ground-truth ``has_breaking_change`` flag.

Usage::

    uv run python -m evals.runner

Add labeled examples as JSON files under ``evals/data/`` first.
See ``evals/README.md`` for the expected schema.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow running as ``python -m evals.runner`` from the project root without
# installing the package (src layout).
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# The table uses box-drawing (─) and check (✓/✗) glyphs; force UTF-8 so output
# doesn't crash on Windows consoles that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from evals.loader import EvalExample, load_examples


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def _predict(example: EvalExample) -> bool:
    """Run extract_breaking_changes on stored evidence; return True if breaking."""
    from ops_agent.state import EvidenceItem, RenovateReviewState
    from ops_agent.graphs.renovate_review import extract_breaking_changes

    evidence = [
        EvidenceItem(source=e.source, url=e.url, text=e.text)
        for e in example.evidence
    ]

    # Build a minimal state — only the fields extract_breaking_changes reads.
    state: RenovateReviewState = {  # type: ignore[typeddict-item]
        "pr_index": 0,
        "dependency": example.dependency,
        "current_version": example.old_version,
        "new_version": example.new_version,
        "diff": "",
        "renovate_rating": None,
        "evidence": evidence,
        "verdict": None,
        "posted": False,
    }

    result: dict[str, Any] = extract_breaking_changes(state)
    findings = result.get("_findings", [])
    return len(findings) > 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    labels: list[bool], preds: list[bool]
) -> dict[str, float]:
    """Compute precision, recall, and F1 for the positive (breaking) class."""
    tp = sum(1 for l, p in zip(labels, preds) if l and p)
    fp = sum(1 for l, p in zip(labels, preds) if not l and p)
    fn = sum(1 for l, p in zip(labels, preds) if l and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_COL_W = [30, 10, 10, 8, 10]
_HEADERS = ["PR ref", "ground truth", "predicted", "correct", "dependency"]


def _row(cells: list[str]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, _COL_W))


def _print_table(examples: list[EvalExample], preds: list[bool]) -> None:
    print()
    print(_row(_HEADERS))
    print(_row(["─" * w for w in _COL_W]))
    for ex, pred in zip(examples, preds):
        gt_str = "breaking" if ex.has_breaking_change else "clear"
        pr_str = "breaking" if pred else "clear"
        correct = "✓" if ex.has_breaking_change == pred else "✗"
        print(_row([ex.pr_ref[:28], gt_str, pr_str, correct, ex.dependency[:28]]))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    examples = load_examples()

    if not examples:
        print("no eval data yet — add labeled JSON files to evals/data/")
        sys.exit(0)

    print(f"Running evals on {len(examples)} example(s)…")

    labels: list[bool] = []
    preds: list[bool] = []
    errors: list[str] = []

    for ex in examples:
        if not ex.evidence:
            print(f"  [SKIP] {ex.pr_ref}: no stored evidence — skipping (add evidence to run extraction)")
            continue
        try:
            pred = _predict(ex)
        except Exception as exc:
            errors.append(f"{ex.pr_ref}: {exc}")
            continue
        labels.append(ex.has_breaking_change)
        preds.append(pred)

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")

    if not labels:
        print("No examples could be evaluated (all skipped or errored).")
        sys.exit(0)

    _print_table(
        [ex for ex in examples if ex.evidence],
        preds,
    )

    metrics = _compute_metrics(labels, preds)
    print(f"Results ({len(labels)} examples evaluated):")
    print(f"  Precision : {metrics['precision']:.3f}  (TP={metrics['tp']}, FP={metrics['fp']})")
    print(f"  Recall    : {metrics['recall']:.3f}  (TP={metrics['tp']}, FN={metrics['fn']})")
    print(f"  F1        : {metrics['f1']:.3f}")
    print()


if __name__ == "__main__":
    main()
