"""Summarize Fibonacci grokking training logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_step(rows: list[dict[str, str]], metric: str, threshold: float) -> int | None:
    for row in rows:
        if float(row[metric]) >= threshold:
            return int(row["step"])
    return None


def first_step_at_or_below(rows: list[dict[str, str]], metric: str, threshold: float) -> int | None:
    for row in rows:
        if float(row[metric]) <= threshold:
            return int(row["step"])
    return None


def summarize(path: Path, threshold: float) -> dict[str, str | int | float | None]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"empty log: {path}")

    last = rows[-1]
    if "train_acc" not in last and "train_rel_mae" in last:
        fit_threshold = float(last.get("fit_rel_mae", 1e-4))
        gen_threshold = float(last.get("generalization_rel_mae", 1e-2))
        fit_step = first_step_at_or_below(rows, "train_rel_mae", fit_threshold)
        gen_step = first_step_at_or_below(rows, "test_rel_mae", gen_threshold)
        gap = None if fit_step is None or gen_step is None else gen_step - fit_step
        return {
            "path": str(path),
            "model": last.get("model", "unknown"),
            "task": last.get("task", "unknown"),
            "target": last.get("target", "unknown"),
            "train_end": int(last.get("train_end", 0)),
            "test_end": int(last.get("test_end", 0)),
            "fit_threshold": fit_threshold,
            "generalization_threshold": gen_threshold,
            "first_fit_step": fit_step,
            "first_generalization_step": gen_step,
            "grokking_gap": gap,
            "final_train_rel_mae": float(last["train_rel_mae"]),
            "final_test_rel_mae": float(last["test_rel_mae"]),
            "final_rollout_rel_mae": last.get("rollout_rel_mae", ""),
            "final_train_loss": float(last["train_loss"]),
            "final_test_loss": float(last["test_loss"]),
        }

    fit_step = first_step(rows, "train_acc", threshold)
    gen_step = first_step(rows, "test_acc", threshold)
    gap = None if fit_step is None or gen_step is None else gen_step - fit_step
    return {
        "path": str(path),
        "model": last.get("model", "unknown"),
        "task": last.get("task", "unknown"),
        "modulus": int(last.get("modulus", 0)),
        "encoding": last.get("encoding", "unknown"),
        "threshold": threshold,
        "first_fit_step": fit_step,
        "first_generalization_step": gen_step,
        "grokking_gap": gap,
        "final_train_acc": float(last["train_acc"]),
        "final_test_acc": float(last["test_acc"]),
        "final_train_loss": float(last["train_loss"]),
        "final_test_loss": float(last["test_loss"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    summaries = [summarize(path, args.threshold) for path in args.logs]
    fields = list(summaries[0].keys())
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
