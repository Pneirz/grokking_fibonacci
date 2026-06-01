"""Plot training curves from Fibonacci grokking CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_series(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty log: {path}")
    return {
        "step": [float(row["step"]) for row in rows],
        "train_acc": [float(row["train_acc"]) for row in rows],
        "test_acc": [float(row["test_acc"]) for row in rows],
        "train_loss": [float(row["train_loss"]) for row in rows],
        "test_loss": [float(row["test_loss"]) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/curves.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for path in args.logs:
        series = read_series(path)
        label = path.stem
        axes[0].plot(series["step"], series["train_acc"], label=f"{label} train")
        axes[0].plot(series["step"], series["test_acc"], linestyle="--", label=f"{label} test")
        axes[1].plot(series["step"], series["train_loss"], label=f"{label} train")
        axes[1].plot(series["step"], series["test_loss"], linestyle="--", label=f"{label} test")

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Cross-Entropy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

