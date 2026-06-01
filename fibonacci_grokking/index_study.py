"""Index-only raw Fibonacci extrapolation study.

The only model input is the raw index n. The training target is the raw
Fibonacci value F_n, without log transforms or scale transforms.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.polynomial import Chebyshev

from .models import build_model
from .raw_extrapolation import fibonacci_numbers


@dataclass(frozen=True)
class StudyConfig:
    start_index: int = 10
    train_end: int = 30
    near_end: int = 35
    mid_end: int = 45
    far_end: int = 60
    models: str = (
        "poly_raw,linear_raw,kan_raw,mlp_raw_matched_kan,"
        "kan_raw_medium,mlp_raw_matched_kan_medium,"
        "kan_raw_wide,mlp_raw_matched_kan_wide,"
        "efficient_kan_raw,mlp_raw_matched_efficient_kan,"
        "efficient_kan_raw_medium,mlp_raw_matched_efficient_kan_medium,"
        "efficient_kan_raw_wide,mlp_raw_matched_efficient_kan_wide,"
        "mlp_raw_2x512,mlp_raw_wide,multkan_raw,multkan_raw_medium,"
        "linear_recurrence_raw_2,linear_recurrence_raw_4,"
        "gated_mlp_raw_2x512,mul_mlp_raw_2x256,"
        "fourier_mlp_raw,siren_raw,nac_raw,nalu_raw,"
        "iter_rnn_raw_8,iter_gru_raw_8,iter_lstm_raw_8"
    )
    seeds: str = "0,1,2,3,4"
    steps: int = 50_000
    eval_every: int = 500
    loss: str = "mse"
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.0
    scheduler: str = "none"
    min_lr_ratio: float = 0.1
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 50
    curriculum: str = ""
    curriculum_steps: int = 0
    hidden_dim: int = 128
    depth: int = 3
    activation: str = "silu"
    kan_hidden_dim: int = 5
    kan_grid: int = 5
    kan_k: int = 3
    fit_rel_mae: float = 1e-2
    extrap_rel_mae: float = 5e-2
    poly_degree: int = 20
    dtype: str = "float64"
    device: str = "auto"
    output_dir: str = "outputs/index_study"


@dataclass(frozen=True)
class IndexData:
    x_all: np.ndarray
    n_all: np.ndarray
    fib_all: np.ndarray
    scale: float
    train_mask: np.ndarray
    bands: dict[str, np.ndarray]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(config: StudyConfig) -> IndexData:
    if not 0 <= config.start_index < config.train_end:
        raise ValueError("start_index must be smaller than train_end")
    if not config.train_end < config.near_end < config.mid_end < config.far_end:
        raise ValueError("expected train_end < near_end < mid_end < far_end")

    fib = fibonacci_numbers(config.far_end)
    n = np.arange(config.far_end + 1, dtype=np.float64)
    x = n.astype(np.float32).reshape(-1, 1)
    train_mask = (n >= config.start_index) & (n <= config.train_end)
    bands = {
        "train": train_mask,
        "near": (n > config.train_end) & (n <= config.near_end),
        "mid": (n > config.near_end) & (n <= config.mid_end),
        "far": (n > config.mid_end) & (n <= config.far_end),
        "future_all": n > config.train_end,
    }
    return IndexData(
        x_all=x,
        n_all=n,
        fib_all=fib.astype(np.float64),
        scale=float(fib[config.train_end]),
        train_mask=train_mask,
        bands=bands,
    )


def mask_for_train_end(data: IndexData, start_index: int, train_end: int) -> np.ndarray:
    return (data.n_all >= start_index) & (data.n_all <= train_end)


def parse_curriculum(value: str, train_end: int) -> list[int]:
    if not value.strip():
        return [train_end]
    stages = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not stages or stages[-1] != train_end:
        stages.append(train_end)
    if any(stage <= 0 for stage in stages):
        raise ValueError("curriculum stages must be positive")
    if any(left >= right for left, right in zip(stages, stages[1:])):
        raise ValueError("curriculum stages must be strictly increasing")
    if stages[-1] != train_end:
        raise ValueError("curriculum must end at train_end")
    return stages


def curriculum_train_end_for_step(config: StudyConfig, step: int) -> int:
    stages = parse_curriculum(config.curriculum, config.train_end)
    if len(stages) == 1 or config.curriculum_steps <= 0:
        return config.train_end
    stage_idx = min(len(stages) - 1, step // config.curriculum_steps)
    return stages[stage_idx]


def target_for_model(model_name: str) -> str:
    if model_name.endswith("_log") or model_name == "log_linear":
        return "log1p"
    if "_raw" in model_name or model_name in {"poly_raw", "linear_raw"}:
        return "raw"
    raise ValueError(f"cannot infer target for model: {model_name}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def mlp_width_for_one_hidden_layer(target_params: int) -> int:
    """Return h for a one-hidden-layer MLP with about target_params parameters.

    For input dimension 1 and output dimension 1, parameter count is:
    input layer: 1*h + h
    output layer: h*1 + 1
    total = 3h + 1
    """
    return max(1, round((target_params - 1) / 3))


def kan_spec_for_model(model_name: str, config: StudyConfig) -> tuple[int, int]:
    if model_name in {
        "kan_raw",
        "kan_log",
        "multkan_raw",
        "mlp_raw_matched_kan",
        "efficient_kan_raw",
        "mlp_raw_matched_efficient_kan",
    }:
        return config.kan_hidden_dim, config.kan_grid
    if model_name in {
        "kan_raw_medium",
        "multkan_raw_medium",
        "mlp_raw_matched_kan_medium",
        "efficient_kan_raw_medium",
        "mlp_raw_matched_efficient_kan_medium",
    }:
        return max(10, config.kan_hidden_dim * 2), max(8, config.kan_grid)
    if model_name in {
        "kan_raw_wide",
        "multkan_raw_wide",
        "mlp_raw_matched_kan_wide",
        "efficient_kan_raw_wide",
        "mlp_raw_matched_efficient_kan_wide",
    }:
        return max(16, config.kan_hidden_dim * 3), max(10, config.kan_grid)
    raise ValueError(f"no KAN spec for model: {model_name}")


def encode_targets(fib: np.ndarray, target: str, scale: float) -> np.ndarray:
    if target == "log1p":
        return np.log1p(fib).astype(np.float32).reshape(-1, 1)
    if target == "raw":
        return fib.astype(np.float64).reshape(-1, 1)
    if target == "raw_scaled":
        return (fib / scale).astype(np.float32).reshape(-1, 1)
    raise ValueError(f"unknown target: {target}")


def decode_predictions(pred: np.ndarray, target: str, scale: float) -> np.ndarray:
    values = np.asarray(pred, dtype=np.float64).reshape(-1)
    if target == "log1p":
        return np.expm1(np.clip(values, -100.0, 80.0))
    if target == "raw":
        return values
    if target == "raw_scaled":
        return values * scale
    raise ValueError(f"unknown target: {target}")


def band_metrics(pred_raw: np.ndarray, true_raw: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    pred = pred_raw[mask]
    true = true_raw[mask]
    abs_error = np.abs(pred - true)
    rel = np.abs(pred - true) / np.maximum(np.abs(true), 1.0)
    clipped_pred = np.maximum(pred, 0.0)
    log_abs = np.abs(np.log1p(clipped_pred) - np.log1p(true))
    rounded = np.rint(pred)
    return {
        "mean_abs_ae": float(np.mean(abs_error)),
        "mean_rel_ae": float(np.mean(rel)),
        "median_rel_ae": float(np.median(rel)),
        "max_rel_ae": float(np.max(rel)),
        "mean_log_ae": float(np.mean(log_abs)),
        "rounded_exact_acc": float(np.mean(rounded == true)),
        "negative_predictions": int(np.sum(pred < 0.0)),
        "monotonicity_violations": int(np.sum(np.diff(pred) < 0.0)),
    }


def fit_log_linear(data: IndexData) -> np.ndarray:
    x_train = data.x_all[data.train_mask, 0].astype(np.float64)
    y_train = np.log1p(data.fib_all[data.train_mask])
    slope, intercept = np.polyfit(x_train, y_train, deg=1)
    pred_log = slope * data.x_all[:, 0].astype(np.float64) + intercept
    return decode_predictions(pred_log, "log1p", data.scale)


def fit_poly_raw(data: IndexData, degree: int) -> np.ndarray:
    x_train = data.x_all[data.train_mask, 0].astype(np.float64)
    y_train = data.fib_all[data.train_mask]
    deg = min(degree, len(x_train) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Chebyshev.fit(x_train, y_train, deg=deg)
    pred_scaled = model(data.x_all[:, 0].astype(np.float64))
    return decode_predictions(pred_scaled, "raw", data.scale)


def make_neural_model(
    model_name: str,
    config: StudyConfig,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    if model_name in {"linear_log", "linear_raw"}:
        return build_model(
            model_name="mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=1,
            depth=0,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name.startswith("mlp_raw_matched_kan") or model_name.startswith(
        "mlp_raw_matched_efficient_kan"
    ):
        kan_hidden_dim, kan_grid = kan_spec_for_model(model_name, config)
        reference_name = (
            "efficient_kan"
            if model_name.startswith("mlp_raw_matched_efficient_kan")
            else "pykan"
        )
        reference = build_model(
            model_name=reference_name,
            input_dim=1,
            output_dim=1,
            hidden_dim=kan_hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
        matched_hidden = mlp_width_for_one_hidden_layer(count_parameters(reference))
        del reference
        return build_model(
            model_name="mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=matched_hidden,
            depth=1,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name in {"mlp_raw", "mlp_log"}:
        return build_model(
            model_name="mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "mlp_raw_2x512":
        return build_model(
            model_name="mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=512,
            depth=2,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "mlp_raw_wide":
        return build_model(
            model_name="mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=max(256, config.hidden_dim * 2),
            depth=max(4, config.depth),
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=config.start_index / config.train_end,
            kan_grid_max=1.0,
            seed=seed,
            device=device,
        )
    if model_name == "fourier_mlp_raw":
        return build_model(
            model_name="fourier_mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "siren_raw":
        return build_model(
            model_name="siren",
            input_dim=1,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "nac_raw":
        return build_model(
            model_name="nac",
            input_dim=1,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "nalu_raw":
        return build_model(
            model_name="nalu",
            input_dim=1,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name.startswith("iter_rnn_raw"):
        hidden_dim = int(model_name.rsplit("_", maxsplit=1)[-1])
        return build_model(
            model_name="iter_rnn",
            input_dim=1,
            output_dim=1,
            hidden_dim=hidden_dim,
            depth=0,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name.startswith("iter_gru_raw"):
        hidden_dim = int(model_name.rsplit("_", maxsplit=1)[-1])
        return build_model(
            model_name="iter_gru",
            input_dim=1,
            output_dim=1,
            hidden_dim=hidden_dim,
            depth=0,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name.startswith("iter_lstm_raw"):
        hidden_dim = int(model_name.rsplit("_", maxsplit=1)[-1])
        return build_model(
            model_name="iter_lstm",
            input_dim=1,
            output_dim=1,
            hidden_dim=hidden_dim,
            depth=0,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "gated_mlp_raw":
        return build_model(
            model_name="gated_mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "gated_mlp_raw_2x512":
        return build_model(
            model_name="gated_mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=512,
            depth=2,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "mul_mlp_raw":
        return build_model(
            model_name="multiplicative_mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "mul_mlp_raw_2x256":
        return build_model(
            model_name="multiplicative_mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=256,
            depth=2,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "mul_mlp_raw_2x512":
        return build_model(
            model_name="multiplicative_mlp",
            input_dim=1,
            output_dim=1,
            hidden_dim=512,
            depth=2,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name.startswith("linear_recurrence_raw"):
        if model_name == "linear_recurrence_raw":
            hidden_dim = config.hidden_dim
        else:
            hidden_dim = int(model_name.rsplit("_", maxsplit=1)[-1])
        return build_model(
            model_name="linear_recurrence",
            input_dim=1,
            output_dim=1,
            hidden_dim=hidden_dim,
            depth=0,
            activation=config.activation,
            kan_grid=config.kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name in {"kan_raw", "kan_log"}:
        kan_hidden_dim, kan_grid = kan_spec_for_model(model_name, config)
        return build_model(
            model_name="pykan",
            input_dim=1,
            output_dim=1,
            hidden_dim=kan_hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name.startswith("multkan_raw"):
        kan_hidden_dim, kan_grid = kan_spec_for_model(model_name, config)
        return build_model(
            model_name="pykan_mult",
            input_dim=1,
            output_dim=1,
            hidden_dim=kan_hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "kan_raw_medium":
        return build_model(
            model_name="pykan",
            input_dim=1,
            output_dim=1,
            hidden_dim=max(10, config.kan_hidden_dim * 2),
            depth=config.depth,
            activation=config.activation,
            kan_grid=max(8, config.kan_grid),
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name == "kan_raw_wide":
        return build_model(
            model_name="pykan",
            input_dim=1,
            output_dim=1,
            hidden_dim=max(16, config.kan_hidden_dim * 3),
            depth=config.depth,
            activation=config.activation,
            kan_grid=max(10, config.kan_grid),
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    if model_name.startswith("efficient_kan_raw"):
        kan_hidden_dim, kan_grid = kan_spec_for_model(model_name, config)
        return build_model(
            model_name="efficient_kan",
            input_dim=1,
            output_dim=1,
            hidden_dim=kan_hidden_dim,
            depth=config.depth,
            activation=config.activation,
            kan_grid=kan_grid,
            kan_k=config.kan_k,
            kan_grid_min=float(config.start_index),
            kan_grid_max=float(config.train_end),
            seed=seed,
            device=device,
        )
    raise ValueError(f"unknown neural model: {model_name}")


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"unknown dtype: {dtype}")


def cast_model_dtype(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    if dtype == torch.float64:
        return model.double()
    if dtype == torch.float32:
        return model.float()
    raise ValueError(f"unsupported model dtype: {dtype}")


@torch.no_grad()
def predict_neural(
    model: torch.nn.Module,
    x_all: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    model.eval()
    x = torch.tensor(x_all, dtype=dtype, device=device)
    return model(x).detach().cpu().numpy().reshape(-1)


def relative_mae(pred_raw: np.ndarray, true_raw: np.ndarray, mask: np.ndarray) -> float:
    pred = pred_raw[mask]
    true = true_raw[mask]
    return float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 1.0)))


def raw_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "mae":
        return F.l1_loss(pred, target)
    if loss_name == "huber":
        return F.smooth_l1_loss(pred, target, beta=1.0)
    raise ValueError(f"unknown loss: {loss_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: StudyConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.steps),
            eta_min=config.lr * config.min_lr_ratio,
        )
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, config.steps // 4),
            gamma=max(0.0, min(1.0, config.min_lr_ratio)),
        )
    raise ValueError(f"unknown scheduler: {config.scheduler}")


def build_optimizer(
    model: torch.nn.Module,
    model_name: str,
    config: StudyConfig,
) -> torch.optim.Optimizer:
    weight_decay = config.weight_decay if model_name not in {"linear_log", "linear_raw"} else 0.0
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=weight_decay)
    if config.optimizer == "adamw_amsgrad":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )
    if config.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=weight_decay)
    if config.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=weight_decay)
    if config.optimizer == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=config.lr, weight_decay=weight_decay)
    if config.optimizer == "rprop":
        return torch.optim.Rprop(model.parameters(), lr=config.lr)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=weight_decay)
    if config.optimizer == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=config.lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def train_neural(
    model_name: str,
    seed: int,
    data: IndexData,
    config: StudyConfig,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, int | float | str]]]:
    set_seed(seed)
    target = target_for_model(model_name)
    y_all = encode_targets(data.fib_all, target, data.scale)
    torch_dtype = resolve_torch_dtype(config.dtype)
    x_all_t = torch.tensor(data.x_all, dtype=torch_dtype, device=device)
    y_all_t = torch.tensor(y_all, dtype=torch_dtype, device=device)

    model = cast_model_dtype(make_neural_model(model_name, config, seed, device), torch_dtype)
    n_params = count_parameters(model)
    optimizer = build_optimizer(model, model_name, config)
    scheduler = build_scheduler(optimizer, config)

    curves: list[dict[str, int | float | str]] = []
    first_fit_step: int | None = None
    first_extrap_step: int | None = None
    start_time = time.perf_counter()

    for step in range(config.steps + 1):
        if step % config.eval_every == 0 or step == config.steps:
            pred_target = predict_neural(model, data.x_all, device, torch_dtype)
            pred_raw = decode_predictions(pred_target, target, data.scale)
            train_rel = relative_mae(pred_raw, data.fib_all, data.bands["train"])
            near_rel = relative_mae(pred_raw, data.fib_all, data.bands["near"])
            mid_rel = relative_mae(pred_raw, data.fib_all, data.bands["mid"])
            far_rel = relative_mae(pred_raw, data.fib_all, data.bands["far"])
            if first_fit_step is None and train_rel <= config.fit_rel_mae:
                first_fit_step = step
            if first_extrap_step is None and far_rel <= config.extrap_rel_mae:
                first_extrap_step = step
            curves.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "target": target,
                    "n_params": n_params,
                    "dtype": config.dtype,
                    "loss": config.loss,
                    "optimizer": config.optimizer,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "grad_clip": config.grad_clip,
                    "scheduler": config.scheduler,
                    "activation": config.activation,
                    "curriculum": config.curriculum,
                    "curriculum_steps": config.curriculum_steps,
                    "active_train_end": curriculum_train_end_for_step(config, step),
                    "step": step,
                    "elapsed_sec": round(time.perf_counter() - start_time, 6),
                    "train_rel_ae": train_rel,
                    "near_rel_ae": near_rel,
                    "mid_rel_ae": mid_rel,
                    "far_rel_ae": far_rel,
                    "first_fit_step": "" if first_fit_step is None else first_fit_step,
                    "first_far_extrap_step": ""
                    if first_extrap_step is None
                    else first_extrap_step,
                }
            )

        if step == config.steps:
            break

        model.train()
        active_end = curriculum_train_end_for_step(config, step)
        train_mask = mask_for_train_end(data, config.start_index, active_end)
        train_indices = torch.tensor(np.nonzero(train_mask)[0], dtype=torch.long, device=device)
        x_train = x_all_t.index_select(0, train_indices)
        y_train = y_all_t.index_select(0, train_indices)
        if config.optimizer == "lbfgs":
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss_value = raw_loss(model(x_train), y_train, config.loss)
                loss_value.backward()
                if config.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                return loss_value

            optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = raw_loss(model(x_train), y_train, config.loss)
            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    final_pred_target = predict_neural(model, data.x_all, device, torch_dtype)
    return decode_predictions(final_pred_target, target, data.scale), curves


def summarize_prediction(
    model_name: str,
    seed: int | str,
    target: str,
    n_params: int | str,
    pred_raw: np.ndarray,
    data: IndexData,
    config: StudyConfig,
    elapsed_sec: float,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for band_name, mask in data.bands.items():
        metrics = band_metrics(pred_raw, data.fib_all, mask)
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "target": target,
                "n_params": n_params,
                "dtype": config.dtype,
                "loss": config.loss,
                "optimizer": config.optimizer,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "grad_clip": config.grad_clip,
                "scheduler": config.scheduler,
                "activation": config.activation,
                "curriculum": config.curriculum,
                "curriculum_steps": config.curriculum_steps,
                "band": band_name,
                "start_index": config.start_index,
                "train_end": config.train_end,
                "near_end": config.near_end,
                "mid_end": config.mid_end,
                "far_end": config.far_end,
                "elapsed_sec": elapsed_sec,
                **metrics,
            }
        )
    return rows


def band_for_n(n: float, config: StudyConfig) -> str:
    if config.start_index <= n <= config.train_end:
        return "train"
    if config.train_end < n <= config.near_end:
        return "near"
    if config.near_end < n <= config.mid_end:
        return "mid"
    if config.mid_end < n <= config.far_end:
        return "far"
    return "unused"


def prediction_rows(
    model_name: str,
    seed: int | str,
    target: str,
    pred_raw: np.ndarray,
    data: IndexData,
    config: StudyConfig,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for n, true_value, pred_value in zip(data.n_all, data.fib_all, pred_raw):
        abs_error = abs(float(pred_value) - float(true_value))
        rel_error = abs_error / max(abs(float(true_value)), 1.0)
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "target": target,
                "dtype": config.dtype,
                "loss": config.loss,
                "optimizer": config.optimizer,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "activation": config.activation,
                "curriculum": config.curriculum,
                "curriculum_steps": config.curriculum_steps,
                "n": int(n),
                "band": band_for_n(float(n), config),
                "true_raw": float(true_value),
                "pred_raw": float(pred_value),
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    q1 = float(np.quantile(ordered, 0.25))
    q3 = float(np.quantile(ordered, 0.75))
    return float(median), q1, q3


def write_report(
    path: Path,
    result_rows: list[dict[str, int | float | str]],
    curve_rows: list[dict[str, int | float | str]],
    config: StudyConfig,
) -> None:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in result_rows:
        if row["band"] in {"near", "mid", "far"}:
            key = (str(row["model"]), str(row["band"]))
            groups.setdefault(key, []).append(float(row["mean_rel_ae"]))

    models = parse_csv(config.models)
    bands = ["near", "mid", "far"]
    lines = [
        "# Index-Only Fibonacci Extrapolation Report",
        "",
        "Input feature: raw scalar `n` only.",
        "",
        f"Train range: `n={config.start_index}..{config.train_end}`.",
        f"Near: `{config.train_end + 1}..{config.near_end}`; "
        f"mid: `{config.near_end + 1}..{config.mid_end}`; "
        f"far: `{config.mid_end + 1}..{config.far_end}`.",
        "",
        "Primary neural targets are raw Fibonacci values: "
        "`F_n`. Any scaled or log-space model is an optional diagnostic, not "
        "part of the raw primary comparison.",
        "",
        "## Median Mean Relative Absolute Error",
        "",
        "| Model | Near | Mid | Far |",
        "|---|---:|---:|---:|",
    ]
    for model in models:
        cells = []
        for band in bands:
            values = groups.get((model, band), [])
            if values:
                med, q1, q3 = median_iqr(values)
                cells.append(f"{med:.6g} [{q1:.6g}, {q3:.6g}]")
            else:
                cells.append("n/a")
        lines.append(f"| `{model}` | {cells[0]} | {cells[1]} | {cells[2]} |")

    params_by_model: dict[str, set[str]] = {}
    for row in result_rows:
        params_by_model.setdefault(str(row["model"]), set()).add(str(row["n_params"]))
    lines.extend(
        [
            "",
            "## Parameter Counts",
            "",
            "| Model | Trainable parameters |",
            "|---|---:|",
        ]
    )
    for model in models:
        values = sorted(params_by_model.get(model, {"n/a"}))
        lines.append(f"| `{model}` | {', '.join(values)} |")

    lines.extend(
        [
            "",
            "## Learning Events",
            "",
            f"Train fit threshold: `{config.fit_rel_mae}` mean relative error.",
            f"Far extrapolation threshold: `{config.extrap_rel_mae}` mean relative error.",
            "",
            "| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    final_by_run: dict[tuple[str, str], dict[str, int | float | str]] = {}
    best_far_by_run: dict[tuple[str, str], dict[str, int | float | str]] = {}
    for row in curve_rows:
        key = (str(row["model"]), str(row["seed"]))
        final_by_run[key] = row
        if key not in best_far_by_run or float(row["far_rel_ae"]) < float(
            best_far_by_run[key]["far_rel_ae"]
        ):
            best_far_by_run[key] = row

    if final_by_run:
        for (model, seed), row in sorted(final_by_run.items()):
            fit = row["first_fit_step"]
            extrap = row["first_far_extrap_step"]
            if fit == "" or extrap == "":
                gap = ""
            else:
                gap = int(extrap) - int(fit)
            lines.append(
                f"| `{model}` | {seed} | {fit} | {extrap} | {gap} | "
                f"{float(row['train_rel_ae']):.6g} | {float(row['far_rel_ae']):.6g} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Best Far Checkpoints",
            "",
            "| Model | Seed | Best far step | Best far | Train at best far | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    if best_far_by_run:
        for (model, seed), row in sorted(best_far_by_run.items()):
            final = final_by_run[(model, seed)]
            fit = final["first_fit_step"]
            extrap = final["first_far_extrap_step"]
            final_train = float(final["train_rel_ae"])
            final_far = float(final["far_rel_ae"])
            if fit == "":
                status = "optimization_failure"
            elif final_train > config.fit_rel_mae:
                status = "unstable_train_fit"
            elif extrap == "":
                status = "post_fit_no_far_success"
            elif int(extrap) > int(fit) and final_far <= config.extrap_rel_mae:
                status = "grokking_like"
            elif final_far <= config.extrap_rel_mae:
                status = "successful_not_delayed"
            else:
                status = "early_or_unstable_far_success"
            lines.append(
                f"| `{model}` | {seed} | {row['step']} | "
                f"{float(row['far_rel_ae']):.6g} | "
                f"{float(row['train_rel_ae']):.6g} | {status} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Reading The Table",
            "",
            "The log-linear baseline is the adversarial ceiling: a model that cannot "
            "beat or approach it has not discovered the asymptotic exponential law "
            "(include `log_linear` explicitly when this reference is needed). The "
            "polynomial baseline is the prefix-fit trap: low train error with bad "
            "future error is evidence against interpreting train fit as grokking.",
            "",
            "A run is treated as meaningful no-grokking evidence only after the "
            "train-fit threshold is reached. Runs without train fit are labeled "
            "`optimization_failure` rather than no-grokking evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_study(config: StudyConfig) -> None:
    output_dir = Path(config.output_dir)
    data = make_data(config)
    device = resolve_device(config.device)

    result_rows: list[dict[str, int | float | str]] = []
    curve_rows: list[dict[str, int | float | str]] = []
    pred_rows: list[dict[str, int | float | str]] = []

    models = parse_csv(config.models)
    seeds = [int(seed) for seed in parse_csv(config.seeds)]

    for model_name in models:
        if model_name == "log_linear":
            start = time.perf_counter()
            pred = fit_log_linear(data)
            result_rows.extend(
                summarize_prediction(
                    model_name,
                    "analytic",
                    "log1p",
                    "analytic",
                    pred,
                    data,
                    config,
                    time.perf_counter() - start,
                )
            )
            pred_rows.extend(prediction_rows(model_name, "analytic", "log1p", pred, data, config))
            print(f"{model_name}: done")
            continue

        if model_name == "poly_raw":
            start = time.perf_counter()
            pred = fit_poly_raw(data, config.poly_degree)
            result_rows.extend(
                summarize_prediction(
                    model_name,
                    "analytic",
                    "raw",
                    "analytic",
                    pred,
                    data,
                    config,
                    time.perf_counter() - start,
                )
            )
            pred_rows.extend(prediction_rows(model_name, "analytic", "raw", pred, data, config))
            print(f"{model_name}: done")
            continue

        for seed in seeds:
            start = time.perf_counter()
            pred, curves = train_neural(model_name, seed, data, config, device)
            target = target_for_model(model_name)
            n_params = curves[-1]["n_params"] if curves else ""
            result_rows.extend(
                summarize_prediction(
                    model_name,
                    seed,
                    target,
                    n_params,
                    pred,
                    data,
                    config,
                    time.perf_counter() - start,
                )
            )
            pred_rows.extend(prediction_rows(model_name, seed, target, pred, data, config))
            curve_rows.extend(curves)
            far_rel = [
                row
                for row in result_rows
                if row["model"] == model_name and row["seed"] == seed and row["band"] == "far"
            ][-1]["mean_rel_ae"]
            print(f"{model_name} seed={seed}: far_mean_rel_ae={far_rel:.6g}")

    write_csv(output_dir / "results.csv", result_rows)
    if curve_rows:
        write_csv(output_dir / "curves.csv", curve_rows)
    if pred_rows:
        write_csv(output_dir / "predictions.csv", pred_rows)
    write_report(output_dir / "REPORT.md", result_rows, curve_rows, config)
    print(f"wrote {output_dir / 'results.csv'}")
    print(f"wrote {output_dir / 'REPORT.md'}")


def parse_args() -> StudyConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-index", type=int, default=StudyConfig.start_index)
    parser.add_argument("--train-end", type=int, default=StudyConfig.train_end)
    parser.add_argument("--near-end", type=int, default=StudyConfig.near_end)
    parser.add_argument("--mid-end", type=int, default=StudyConfig.mid_end)
    parser.add_argument("--far-end", type=int, default=StudyConfig.far_end)
    parser.add_argument("--models", default=StudyConfig.models)
    parser.add_argument("--seeds", default=StudyConfig.seeds)
    parser.add_argument("--steps", type=int, default=StudyConfig.steps)
    parser.add_argument("--eval-every", type=int, default=StudyConfig.eval_every)
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], default=StudyConfig.loss)
    parser.add_argument(
        "--optimizer",
        choices=[
            "adamw",
            "adamw_amsgrad",
            "rmsprop",
            "adagrad",
            "adamax",
            "rprop",
            "sgd",
            "lbfgs",
        ],
        default=StudyConfig.optimizer,
    )
    parser.add_argument("--lr", type=float, default=StudyConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=StudyConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=StudyConfig.grad_clip)
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default=StudyConfig.scheduler)
    parser.add_argument("--min-lr-ratio", type=float, default=StudyConfig.min_lr_ratio)
    parser.add_argument("--lbfgs-max-iter", type=int, default=StudyConfig.lbfgs_max_iter)
    parser.add_argument("--lbfgs-history-size", type=int, default=StudyConfig.lbfgs_history_size)
    parser.add_argument("--curriculum", default=StudyConfig.curriculum)
    parser.add_argument("--curriculum-steps", type=int, default=StudyConfig.curriculum_steps)
    parser.add_argument("--hidden-dim", type=int, default=StudyConfig.hidden_dim)
    parser.add_argument("--depth", type=int, default=StudyConfig.depth)
    parser.add_argument(
        "--activation",
        choices=[
            "gelu",
            "relu",
            "silu",
            "tanh",
            "sigmoid",
            "softplus",
            "elu",
            "leaky_relu",
        ],
        default=StudyConfig.activation,
    )
    parser.add_argument("--kan-hidden-dim", type=int, default=StudyConfig.kan_hidden_dim)
    parser.add_argument("--kan-grid", type=int, default=StudyConfig.kan_grid)
    parser.add_argument("--kan-k", type=int, default=StudyConfig.kan_k)
    parser.add_argument("--fit-rel-mae", type=float, default=StudyConfig.fit_rel_mae)
    parser.add_argument("--extrap-rel-mae", type=float, default=StudyConfig.extrap_rel_mae)
    parser.add_argument("--poly-degree", type=int, default=StudyConfig.poly_degree)
    parser.add_argument("--dtype", choices=["float32", "float64"], default=StudyConfig.dtype)
    parser.add_argument("--device", default=StudyConfig.device)
    parser.add_argument("--output-dir", default=StudyConfig.output_dir)
    args = parser.parse_args()
    return StudyConfig(**vars(args))


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
