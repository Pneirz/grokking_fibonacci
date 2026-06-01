"""Train MLP or PyKAN models on Fibonacci modular grokking tasks."""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_dataset
from .models import build_model


@dataclass(frozen=True)
class TrainConfig:
    model: str = "mlp"
    task: str = "all_pairs"
    modulus: int = 97
    train_frac: float = 0.4
    encoding: str = "onehot"
    seed: int = 0
    steps: int = 50_000
    eval_every: int = 500
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1.0
    hidden_dim: int = 256
    depth: int = 2
    activation: str = "gelu"
    kan_grid: int = 5
    kan_k: int = 3
    kan_grid_min: float = -1.0
    kan_grid_max: float = 1.0
    device: str = "auto"
    threshold: float = 0.99
    stop_at_test_acc: float = 0.0
    output: str = "outputs/run.csv"


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


def tensorize(x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


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


def run_experiment(config: TrainConfig) -> list[dict[str, int | float | str]]:
    set_seed(config.seed)
    device = resolve_device(config.device)

    dataset = make_dataset(
        task=config.task,
        modulus=config.modulus,
        train_frac=config.train_frac,
        seed=config.seed,
        encoding=config.encoding,
    )
    train_x, train_y = tensorize(dataset.train_x, dataset.train_y, device)
    test_x, test_y = tensorize(dataset.test_x, dataset.test_y, device)

    model = build_model(
        model_name=config.model,
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    rows: list[dict[str, int | float | str]] = []
    start_time = time.perf_counter()
    first_fit_step: int | None = None
    first_generalization_step: int | None = None

    for step in range(config.steps + 1):
        if step % config.eval_every == 0 or step == config.steps:
            train_loss, train_acc = evaluate(model, train_x, train_y)
            test_loss, test_acc = evaluate(model, test_x, test_y)

            if first_fit_step is None and train_acc >= config.threshold:
                first_fit_step = step
            if first_generalization_step is None and test_acc >= config.threshold:
                first_generalization_step = step

            row: dict[str, int | float | str] = {
                **asdict(config),
                **dataset.metadata,
                "device_resolved": str(device),
                "step": step,
                "elapsed_sec": round(time.perf_counter() - start_time, 6),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "first_fit_step": "" if first_fit_step is None else first_fit_step,
                "first_generalization_step": ""
                if first_generalization_step is None
                else first_generalization_step,
            }
            rows.append(row)

            if config.stop_at_test_acc > 0.0 and test_acc >= config.stop_at_test_acc:
                break

        if step == config.steps:
            break

        model.train()
        xb, yb = sample_batch(train_x, train_y, config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()

    write_rows(Path(config.output), rows)
    return rows


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "pykan"], default=TrainConfig.model)
    parser.add_argument("--task", choices=["all_pairs", "trajectory"], default=TrainConfig.task)
    parser.add_argument("--modulus", type=int, default=TrainConfig.modulus)
    parser.add_argument("--train-frac", type=float, default=TrainConfig.train_frac)
    parser.add_argument("--encoding", choices=["onehot", "scalar", "fourier"], default=TrainConfig.encoding)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--depth", type=int, default=TrainConfig.depth)
    parser.add_argument("--activation", choices=["gelu", "relu", "silu"], default=TrainConfig.activation)
    parser.add_argument("--kan-grid", type=int, default=TrainConfig.kan_grid)
    parser.add_argument("--kan-k", type=int, default=TrainConfig.kan_k)
    parser.add_argument("--kan-grid-min", type=float, default=TrainConfig.kan_grid_min)
    parser.add_argument("--kan-grid-max", type=float, default=TrainConfig.kan_grid_max)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--threshold", type=float, default=TrainConfig.threshold)
    parser.add_argument("--stop-at-test-acc", type=float, default=TrainConfig.stop_at_test_acc)
    parser.add_argument("--output", default=TrainConfig.output)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    rows = run_experiment(config)
    last = rows[-1]
    print(
        "done "
        f"model={last['model']} task={last['task']} modulus={last['modulus']} "
        f"step={last['step']} train_acc={last['train_acc']:.4f} test_acc={last['test_acc']:.4f} "
        f"output={config.output}"
    )


if __name__ == "__main__":
    main()
