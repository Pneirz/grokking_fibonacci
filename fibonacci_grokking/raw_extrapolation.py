"""Raw Fibonacci extrapolation experiments for MLP and PyKAN models."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model


@dataclass(frozen=True)
class RawConfig:
    model: str = "mlp"
    task: str = "index"
    target: str = "raw"
    start_index: int = 0
    train_end: int = 30
    test_end: int = 45
    seed: int = 0
    steps: int = 20_000
    eval_every: int = 500
    batch_size: int = 0
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    depth: int = 3
    activation: str = "silu"
    kan_grid: int = 5
    kan_k: int = 3
    kan_grid_min: float = -1.0
    kan_grid_max: float = 1.0
    device: str = "auto"
    fit_rel_mae: float = 1e-4
    generalization_rel_mae: float = 1e-2
    output: str = "outputs/raw_index.csv"


@dataclass(frozen=True)
class RawSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    train_raw_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_raw_y: np.ndarray
    scale: float
    input_dim: int
    metadata: dict[str, int | float | str]


def fibonacci_numbers(max_n: int) -> np.ndarray:
    """Return Fibonacci numbers F_0 through F_max_n as float64 values."""
    if max_n < 2:
        raise ValueError("max_n must be at least 2")
    values = [0, 1]
    for _ in range(2, max_n + 1):
        values.append(values[-1] + values[-2])
    return np.array(values, dtype=np.float64)


def make_transition_split(train_end: int, test_end: int, start_index: int) -> RawSplit:
    """Predict raw F_{n+2} from raw (F_n, F_{n+1}) outside the train range."""
    if train_end < 4:
        raise ValueError("train_end must be at least 4")
    if test_end <= train_end + 1:
        raise ValueError("test_end must be at least train_end + 2")
    if start_index < 0 or start_index > train_end - 2:
        raise ValueError("start_index must be between 0 and train_end - 2")

    fib = fibonacci_numbers(test_end)
    scale = float(fib[train_end])
    xs: list[list[float]] = []
    ys: list[float] = []
    raw_ys: list[float] = []
    target_indices: list[int] = []
    for n in range(start_index, test_end - 1):
        target_idx = n + 2
        xs.append([fib[n] / scale, fib[n + 1] / scale])
        ys.append(fib[target_idx] / scale)
        raw_ys.append(fib[target_idx])
        target_indices.append(target_idx)

    x = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)
    raw_y = np.array(raw_ys, dtype=np.float64).reshape(-1, 1)
    idx = np.array(target_indices)
    train_mask = idx <= train_end
    test_mask = idx > train_end

    return RawSplit(
        train_x=x[train_mask],
        train_y=y[train_mask],
        train_raw_y=raw_y[train_mask],
        test_x=x[test_mask],
        test_y=y[test_mask],
        test_raw_y=raw_y[test_mask],
        scale=scale,
        input_dim=2,
        metadata={
            "task": "transition",
            "target": "raw_scaled",
            "start_index": start_index,
            "train_end": train_end,
            "test_end": test_end,
            "scale": scale,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        },
    )


def make_index_split(train_end: int, test_end: int, target: str, start_index: int) -> RawSplit:
    """Predict raw F_n directly from n, optionally in log1p target space."""
    if train_end < 4:
        raise ValueError("train_end must be at least 4")
    if test_end <= train_end:
        raise ValueError("test_end must be greater than train_end")
    if start_index < 0 or start_index > train_end:
        raise ValueError("start_index must be between 0 and train_end")

    fib = fibonacci_numbers(test_end)
    scale = float(fib[train_end])
    n = np.arange(0, test_end + 1, dtype=np.float64)
    x = (n / float(train_end)).astype(np.float32).reshape(-1, 1)
    raw_y = fib.reshape(-1, 1)
    if target == "raw_scaled":
        y = (raw_y / scale).astype(np.float32)
    elif target == "raw":
        y = raw_y.astype(np.float32)
    elif target == "log1p":
        y = np.log1p(raw_y).astype(np.float32)
    else:
        raise ValueError(f"unknown target for index task: {target}")

    train_mask = (n >= start_index) & (n <= train_end)
    test_mask = n > train_end
    return RawSplit(
        train_x=x[train_mask],
        train_y=y[train_mask],
        train_raw_y=raw_y[train_mask],
        test_x=x[test_mask],
        test_y=y[test_mask],
        test_raw_y=raw_y[test_mask],
        scale=scale,
        input_dim=1,
        metadata={
            "task": "index",
            "target": target,
            "start_index": start_index,
            "train_end": train_end,
            "test_end": test_end,
            "scale": scale,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        },
    )


def make_split(config: RawConfig) -> RawSplit:
    if config.task == "transition":
        if config.target != "raw_scaled":
            raise ValueError("transition task currently supports only --target raw_scaled")
        return make_transition_split(config.train_end, config.test_end, config.start_index)
    if config.task == "index":
        return make_index_split(config.train_end, config.test_end, config.target, config.start_index)
    raise ValueError(f"unknown raw task: {config.task}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def decode_predictions(values: torch.Tensor, target: str, scale: float) -> torch.Tensor:
    if target == "raw":
        return values
    if target == "raw_scaled":
        return values * scale
    if target == "log1p":
        return torch.expm1(values)
    raise ValueError(f"unknown target: {target}")


def relative_mae(pred_raw: torch.Tensor, y_raw: torch.Tensor) -> float:
    denom = torch.clamp(torch.abs(y_raw), min=1.0)
    return (torch.abs(pred_raw - y_raw) / denom).mean().item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    raw_y: torch.Tensor,
    target: str,
    scale: float,
) -> tuple[float, float]:
    model.eval()
    pred = model(x)
    loss = F.mse_loss(pred, y).item()
    raw_pred = decode_predictions(pred, target, scale)
    return loss, relative_mae(raw_pred, raw_y)


@torch.no_grad()
def rollout_relative_mae(
    model: torch.nn.Module,
    train_end: int,
    test_end: int,
    scale: float,
    device: torch.device,
) -> float:
    """Roll the transition model forward using its own predictions."""
    model.eval()
    fib = fibonacci_numbers(test_end)
    pred = [fib[0] / scale, fib[1] / scale]
    for _ in range(test_end - 1):
        x = torch.tensor([[pred[-2], pred[-1]]], dtype=torch.float32, device=device)
        next_value = model(x).reshape(-1)[0].item()
        if not math.isfinite(next_value):
            return float("inf")
        pred.append(next_value)

    pred_raw = torch.tensor(pred, dtype=torch.float64, device=device).reshape(-1, 1) * scale
    true_raw = torch.tensor(fib, dtype=torch.float64, device=device).reshape(-1, 1)
    future = slice(train_end + 1, test_end + 1)
    return relative_mae(pred_raw[future], true_raw[future])


def sample_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0 or batch_size >= x.shape[0]:
        return x, y
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device)
    return x[idx], y[idx]


def write_rows(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: RawConfig) -> list[dict[str, int | float | str]]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    split = make_split(config)

    train_x = torch.tensor(split.train_x, dtype=torch.float32, device=device)
    train_y = torch.tensor(split.train_y, dtype=torch.float32, device=device)
    train_raw_y = torch.tensor(split.train_raw_y, dtype=torch.float32, device=device)
    test_x = torch.tensor(split.test_x, dtype=torch.float32, device=device)
    test_y = torch.tensor(split.test_y, dtype=torch.float32, device=device)
    test_raw_y = torch.tensor(split.test_raw_y, dtype=torch.float32, device=device)

    model = build_model(
        model_name=config.model,
        input_dim=split.input_dim,
        output_dim=1,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        activation=config.activation,
        kan_grid=config.kan_grid,
        kan_k=config.kan_k,
        kan_grid_min=config.kan_grid_min,
        kan_grid_max=config.kan_grid_max,
        seed=config.seed,
        device=device,
    )
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lr,
            max_iter=1,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"unknown optimizer: {config.optimizer}")

    rows: list[dict[str, int | float | str]] = []
    start_time = time.perf_counter()
    first_fit_step: int | None = None
    first_generalization_step: int | None = None

    for step in range(config.steps + 1):
        if step % config.eval_every == 0 or step == config.steps:
            train_loss, train_rel_mae = evaluate(
                model, train_x, train_y, train_raw_y, config.target, split.scale
            )
            test_loss, test_rel_mae = evaluate(
                model, test_x, test_y, test_raw_y, config.target, split.scale
            )
            rollout_rel_mae = (
                rollout_relative_mae(model, config.train_end, config.test_end, split.scale, device)
                if config.task == "transition"
                else ""
            )

            if first_fit_step is None and train_rel_mae <= config.fit_rel_mae:
                first_fit_step = step
            if (
                first_generalization_step is None
                and test_rel_mae <= config.generalization_rel_mae
            ):
                first_generalization_step = step

            rows.append(
                {
                    **asdict(config),
                    **split.metadata,
                    "device_resolved": str(device),
                    "step": step,
                    "elapsed_sec": round(time.perf_counter() - start_time, 6),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_rel_mae": train_rel_mae,
                    "test_rel_mae": test_rel_mae,
                    "rollout_rel_mae": rollout_rel_mae,
                    "first_fit_step": "" if first_fit_step is None else first_fit_step,
                    "first_generalization_step": ""
                    if first_generalization_step is None
                    else first_generalization_step,
                }
            )

        if step == config.steps:
            break

        model.train()
        xb, yb = sample_batch(train_x, train_y, config.batch_size)
        if config.optimizer == "lbfgs":
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss_value = F.mse_loss(model(train_x), train_y)
                loss_value.backward()
                return loss_value

            optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

    write_rows(Path(config.output), rows)
    return rows


def parse_args() -> RawConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "pykan"], default=RawConfig.model)
    parser.add_argument("--task", choices=["transition", "index"], default=RawConfig.task)
    parser.add_argument("--target", choices=["raw", "raw_scaled", "log1p"], default=RawConfig.target)
    parser.add_argument("--start-index", type=int, default=RawConfig.start_index)
    parser.add_argument("--train-end", type=int, default=RawConfig.train_end)
    parser.add_argument("--test-end", type=int, default=RawConfig.test_end)
    parser.add_argument("--seed", type=int, default=RawConfig.seed)
    parser.add_argument("--steps", type=int, default=RawConfig.steps)
    parser.add_argument("--eval-every", type=int, default=RawConfig.eval_every)
    parser.add_argument("--batch-size", type=int, default=RawConfig.batch_size)
    parser.add_argument("--optimizer", choices=["adamw", "lbfgs"], default=RawConfig.optimizer)
    parser.add_argument("--lr", type=float, default=RawConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=RawConfig.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=RawConfig.hidden_dim)
    parser.add_argument("--depth", type=int, default=RawConfig.depth)
    parser.add_argument("--activation", choices=["gelu", "relu", "silu"], default=RawConfig.activation)
    parser.add_argument("--kan-grid", type=int, default=RawConfig.kan_grid)
    parser.add_argument("--kan-k", type=int, default=RawConfig.kan_k)
    parser.add_argument("--kan-grid-min", type=float, default=RawConfig.kan_grid_min)
    parser.add_argument("--kan-grid-max", type=float, default=RawConfig.kan_grid_max)
    parser.add_argument("--device", default=RawConfig.device)
    parser.add_argument("--fit-rel-mae", type=float, default=RawConfig.fit_rel_mae)
    parser.add_argument(
        "--generalization-rel-mae",
        type=float,
        default=RawConfig.generalization_rel_mae,
    )
    parser.add_argument("--output", default=RawConfig.output)
    args = parser.parse_args()
    return RawConfig(**vars(args))


def main() -> None:
    config = parse_args()
    rows = run_experiment(config)
    last = rows[-1]
    print(
        "done "
        f"model={last['model']} task={last['task']} target={last['target']} "
        f"step={last['step']} train_rel_mae={last['train_rel_mae']:.6g} "
        f"test_rel_mae={last['test_rel_mae']:.6g} "
        f"rollout_rel_mae={last['rollout_rel_mae']} output={config.output}"
    )


if __name__ == "__main__":
    main()
