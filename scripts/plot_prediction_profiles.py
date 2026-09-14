"""Regenerate the final prediction-profile figure from archived CSV outputs.

The figure summarizes the saved final predictions for five representative
models shown in the manuscript: MLP 2x512, Product MLP, MultKAN, NALU, and the
two-dimensional recurrent control. It does not train models or recompute
experiment outputs. Instead, it aggregates the archived ten-seed
``predictions.csv`` files by their per-index medians.

By default the script discovers the confirmatory prediction files under the
repository's ``outputs/`` directory and writes PDF, SVG, and 1200-dpi PNG
versions to ``figures/``. The PDF and SVG are the preferred publication assets;
the PNG is supplied for workflows that require a raster image.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION_FILES = sorted((REPOSITORY_ROOT / "outputs").glob("confirm_*/predictions.csv"))
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "figures"

MODELS = (
    "mlp_raw_2x512",
    "mul_mlp_raw_2x256",
    "multkan_raw_medium",
    "nalu_raw",
    "linear_recurrence_raw_2",
)

LABELS = {
    "mlp_raw_2x512": "MLP 2x512",
    "mul_mlp_raw_2x256": "Product MLP",
    "multkan_raw_medium": "MultKAN",
    "nalu_raw": "NALU",
    "linear_recurrence_raw_2": "Recurrent dim 2",
}

COLORS = {
    "mlp_raw_2x512": "#4e79a7",
    "mul_mlp_raw_2x256": "#f28e2b",
    "multkan_raw_medium": "#59a14f",
    "nalu_raw": "#e377c2",
    "linear_recurrence_raw_2": "#000000",
}


def read_predictions(
    paths: list[Path],
) -> tuple[dict[str, dict[int, list[float]]], dict[str, dict[int, list[float]]], dict[int, float]]:
    """Read raw predictions and relative errors, validating shared Fibonacci targets."""
    predictions: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    relative_errors: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    targets: dict[int, float] = {}

    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                model = row["model"]
                if model not in MODELS:
                    continue
                index = int(row["n"])
                target = float(row["true_raw"])
                previous = targets.setdefault(index, target)
                if previous != target:
                    raise ValueError(f"inconsistent target for n={index} in {path}")
                predictions[model][index].append(float(row["pred_raw"]))
                relative_errors[model][index].append(float(row["rel_error"]))

    missing = [model for model in MODELS if model not in predictions]
    if missing:
        raise ValueError(f"missing prediction rows for: {', '.join(missing)}")
    if sorted(targets) != list(range(0, 61)):
        raise ValueError("expected archived predictions for every index from n=0 through n=60")
    for model in MODELS:
        counts = {len(predictions[model][index]) for index in range(0, 61)}
        if counts != {10}:
            raise ValueError(f"expected ten archived seeds per index for {model}; found {sorted(counts)}")

    return predictions, relative_errors, targets


def signed_log10(values: np.ndarray) -> np.ndarray:
    """Use a sign-preserving logarithm so non-positive predictions remain visible."""
    return np.sign(values) * np.log10(np.abs(values) + 1.0)


def normalize_svg(path: Path) -> None:
    """Remove insignificant line-ending whitespace from a generated SVG."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def plot(paths: list[Path], output_dir: Path, dpi: int) -> list[Path]:
    predictions, relative_errors, targets = read_predictions(paths)
    indices = np.arange(10, 61)
    future_indices = np.arange(31, 61)

    # 122 mm is the manuscript's final figure width. All visible text is at
    # least 6.5 pt at that physical size.
    fig, (prediction_axis, error_axis) = plt.subplots(1, 2, figsize=(122 / 25.4, 2.75))

    truth = np.array([targets[int(index)] for index in indices])
    (truth_line,) = prediction_axis.plot(
        indices,
        signed_log10(truth),
        color="0.3",
        linestyle="--",
        linewidth=1.1,
        label="Ground truth",
        zorder=2,
    )
    model_handles: list[Line2D] = []
    for model in MODELS:
        median_prediction = np.array([np.median(predictions[model][int(index)]) for index in indices])
        (line,) = prediction_axis.plot(
            indices,
            signed_log10(median_prediction),
            color=COLORS[model],
            linewidth=1.45 if model == "linear_recurrence_raw_2" else 1.15,
            label=LABELS[model],
            zorder=3,
        )
        model_handles.append(line)

        median_error = np.array(
            [np.median(relative_errors[model][int(index)]) for index in future_indices]
        )
        error_axis.plot(
            future_indices,
            median_error,
            color=COLORS[model],
            linewidth=1.45 if model == "linear_recurrence_raw_2" else 1.15,
            zorder=3,
        )

    prediction_axis.axvspan(10, 30, color="0.85", alpha=0.45, zorder=0)
    prediction_axis.axvline(30, color="0.35", linestyle="--", linewidth=0.8, zorder=1)
    prediction_axis.set_title("Median decoded prediction", fontsize=8)
    prediction_axis.set_xlabel("Index n", fontsize=7)
    prediction_axis.set_ylabel("signed log10(value + 1)", fontsize=7)
    prediction_axis.set_xlim(9.5, 60.5)

    error_axis.axvspan(31, 35, color="#dce6f2", alpha=0.75, zorder=0)
    error_axis.axvspan(35, 45, color="#f9ecd5", alpha=0.75, zorder=0)
    error_axis.axvspan(45, 60, color="#f7dddd", alpha=0.75, zorder=0)
    threshold_line = error_axis.axhline(
        5e-2,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        label="Far-success threshold",
        zorder=1,
    )
    error_axis.set_title("Median relative error after training", fontsize=8)
    error_axis.set_xlabel("Index n", fontsize=7)
    error_axis.set_ylabel("Relative absolute error", fontsize=7)
    error_axis.set_yscale("log")
    error_axis.set_xlim(30.5, 60.5)
    error_axis.set_ylim(1e-8, 2)

    for axis in (prediction_axis, error_axis):
        axis.tick_params(labelsize=6.5)
        axis.grid(True, which="both", linewidth=0.3, alpha=0.45)

    fig.legend(
        [truth_line, *model_handles, threshold_line],
        ["Ground truth", *(LABELS[model] for model in MODELS), "Far-success threshold"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.87, bottom=0.32, wspace=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "fig_prediction_profiles"
    outputs = [prefix.with_suffix(".pdf"), prefix.with_suffix(".svg"), prefix.with_suffix(".png")]
    fig.savefig(outputs[0], format="pdf")
    fig.savefig(outputs[1], format="svg")
    normalize_svg(outputs[1])
    fig.savefig(outputs[2], format="png", dpi=dpi)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        default=DEFAULT_PREDICTION_FILES,
        help="prediction CSV files; defaults to all archived confirmatory files",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()

    for output in plot(args.predictions, args.output_dir, args.dpi):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
