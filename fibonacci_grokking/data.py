"""Dataset generation for Fibonacci modular grokking experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    input_dim: int
    output_dim: int
    metadata: dict[str, int | float | str]


def fibonacci_mod_sequence(modulus: int, start_a: int = 0, start_b: int = 1) -> list[tuple[int, int]]:
    """Return one Pisano-cycle worth of Fibonacci states modulo `modulus`."""
    if modulus < 2:
        raise ValueError("modulus must be at least 2")

    state = (start_a % modulus, start_b % modulus)
    seen: set[tuple[int, int]] = set()
    states: list[tuple[int, int]] = []
    while state not in seen:
        seen.add(state)
        states.append(state)
        state = (state[1], (state[0] + state[1]) % modulus)
    return states


def encode_pairs(pairs: np.ndarray, modulus: int, encoding: str) -> np.ndarray:
    """Encode pair states as scalar, one-hot, or Fourier features."""
    if encoding == "scalar":
        denom = max(1, modulus - 1)
        return (2.0 * pairs.astype(np.float32) / denom) - 1.0

    if encoding == "onehot":
        out = np.zeros((pairs.shape[0], 2 * modulus), dtype=np.float32)
        rows = np.arange(pairs.shape[0])
        out[rows, pairs[:, 0]] = 1.0
        out[rows, modulus + pairs[:, 1]] = 1.0
        return out

    if encoding == "fourier":
        angles = 2.0 * np.pi * pairs.astype(np.float32) / float(modulus)
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)

    raise ValueError(f"unknown encoding: {encoding}")


def make_pair_dataset(
    modulus: int,
    train_frac: float,
    seed: int,
    encoding: str,
) -> DatasetSplit:
    """Create all modular Fibonacci transition pairs and a random train/test split."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")

    pairs = np.array([(a, b) for a in range(modulus) for b in range(modulus)], dtype=np.int64)
    targets = ((pairs[:, 0] + pairs[:, 1]) % modulus).astype(np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_train = int(round(train_frac * len(pairs)))
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    train_x = encode_pairs(pairs[train_idx], modulus, encoding)
    test_x = encode_pairs(pairs[test_idx], modulus, encoding)

    return DatasetSplit(
        train_x=train_x,
        train_y=targets[train_idx],
        test_x=test_x,
        test_y=targets[test_idx],
        input_dim=train_x.shape[1],
        output_dim=modulus,
        metadata={
            "task": "all_pairs",
            "modulus": modulus,
            "train_frac": train_frac,
            "seed": seed,
            "encoding": encoding,
            "n_train": int(n_train),
            "n_test": int(len(test_idx)),
        },
    )


def make_trajectory_dataset(
    modulus: int,
    train_frac: float,
    seed: int,
    encoding: str,
) -> DatasetSplit:
    """Create a split over the actual Fibonacci trajectory modulo `modulus`."""
    del seed
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")

    pairs = np.array(fibonacci_mod_sequence(modulus), dtype=np.int64)
    targets = ((pairs[:, 0] + pairs[:, 1]) % modulus).astype(np.int64)
    n_train = max(1, min(len(pairs) - 1, int(round(train_frac * len(pairs)))))

    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]
    train_x = encode_pairs(train_pairs, modulus, encoding)
    test_x = encode_pairs(test_pairs, modulus, encoding)

    return DatasetSplit(
        train_x=train_x,
        train_y=targets[:n_train],
        test_x=test_x,
        test_y=targets[n_train:],
        input_dim=train_x.shape[1],
        output_dim=modulus,
        metadata={
            "task": "trajectory",
            "modulus": modulus,
            "train_frac": train_frac,
            "seed": "not_used",
            "encoding": encoding,
            "n_train": int(n_train),
            "n_test": int(len(test_pairs)),
            "period": int(len(pairs)),
        },
    )


def make_dataset(
    task: str,
    modulus: int,
    train_frac: float,
    seed: int,
    encoding: str,
) -> DatasetSplit:
    """Factory for supported Fibonacci grokking datasets."""
    if task == "all_pairs":
        return make_pair_dataset(modulus, train_frac, seed, encoding)
    if task == "trajectory":
        return make_trajectory_dataset(modulus, train_frac, seed, encoding)
    raise ValueError(f"unknown task: {task}")

