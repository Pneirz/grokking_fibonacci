"""Plot short-prefix training dynamics from saved curve CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_curves(path: Path) -> dict[str, dict[int, dict[str, list[float]]]]:
    grouped: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            step = int(row["step"])
            grouped[model][step]["train"].append(float(row["train_rel_ae"]))
            grouped[model][step]["near"].append(float(row["near_rel_ae"]))
            grouped[model][step]["mid"].append(float(row["mid_rel_ae"]))
            grouped[model][step]["far"].append(float(row["far_rel_ae"]))
    return grouped


def merge_curves(paths: list[Path]) -> dict[str, dict[int, dict[str, list[float]]]]:
    merged: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for path in paths:
        curves = read_curves(path)
        for model, by_step in curves.items():
            for step, metrics in by_step.items():
                for metric, values in metrics.items():
                    merged[model][step][metric].extend(values)
    return merged


def plot(paths: list[Path], output: Path) -> None:
    curves = merge_curves(paths)
    labels = {
        "mlp_raw_2x512": "MLP 2x512",
        "mul_mlp_raw_2x256": "Product MLP",
        "multkan_raw_medium": "MultKAN",
        "fourier_mlp_raw": "Fourier MLP",
        "siren_raw": "SIREN",
        "nac_raw": "NAC",
        "nalu_raw": "NALU",
        "iter_rnn_raw_8": "RNN",
        "iter_gru_raw_8": "GRU",
        "iter_lstm_raw_8": "LSTM",
        "linear_recurrence_raw_2": "Recurrent dim 2",
    }
    colors = {
        "mlp_raw_2x512": "#1f77b4",
        "mul_mlp_raw_2x256": "#ff7f0e",
        "multkan_raw_medium": "#2ca02c",
        "fourier_mlp_raw": "#d62728",
        "siren_raw": "#9467bd",
        "nac_raw": "#8c564b",
        "nalu_raw": "#e377c2",
        "iter_rnn_raw_8": "#7f7f7f",
        "iter_gru_raw_8": "#bcbd22",
        "iter_lstm_raw_8": "#17becf",
        "linear_recurrence_raw_2": "#000000",
    }
    model_order = [model for model in labels if model in curves]

    metrics = [
        ("train", "Train error", 1e-2),
        ("near", "Near-band error", 5e-2),
        ("mid", "Mid-band error", 5e-2),
        ("far", "Far-band error", 5e-2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.35), sharex=True)
    flat_axes = axes.ravel()
    handles_by_model = {}
    for model in model_order:
        by_step = curves[model]
        steps = sorted(by_step)
        for ax, (metric, _title, _threshold) in zip(flat_axes, metrics):
            medians = [np.median(by_step[step][metric]) for step in steps]
            q1 = [np.quantile(by_step[step][metric], 0.25) for step in steps]
            q3 = [np.quantile(by_step[step][metric], 0.75) for step in steps]
            (line,) = ax.plot(
                steps,
                medians,
                color=colors.get(model),
                linewidth=1.35,
                label=labels.get(model, model),
            )
            ax.fill_between(steps, q1, q3, color=colors.get(model), alpha=0.12)
            handles_by_model.setdefault(model, line)

    for ax, (_metric, title, threshold) in zip(flat_axes, metrics):
        ax.axhline(threshold, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Updates")
        ax.set_ylabel("Mean relative absolute error")
        ax.set_yscale("log")
        ax.set_xlim(left=0)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.45)
    fig.legend(
        [handles_by_model[model] for model in model_order],
        [labels.get(model, model) for model in model_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=7,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(args.curves, args.output)


if __name__ == "__main__":
    main()
