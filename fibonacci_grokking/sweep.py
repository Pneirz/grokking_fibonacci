"""Run a small grid of Fibonacci grokking experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .train import TrainConfig, run_experiment


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="mlp")
    parser.add_argument("--encodings", default="onehot,scalar,fourier")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--modulus", type=int, default=17)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    args = parser.parse_args()

    base = TrainConfig(
        modulus=args.modulus,
        train_frac=args.train_frac,
        steps=args.steps,
        eval_every=args.eval_every,
    )

    for model in parse_csv(args.models):
        for encoding in parse_csv(args.encodings):
            for seed_text in parse_csv(args.seeds):
                seed = int(seed_text)
                output = args.output_dir / f"{model}_{encoding}_m{args.modulus}_seed{seed}.csv"
                config = replace(
                    base,
                    model=model,
                    encoding=encoding,
                    seed=seed,
                    output=str(output),
                )
                rows = run_experiment(config)
                last = rows[-1]
                print(
                    f"{output}: train_acc={last['train_acc']:.4f} "
                    f"test_acc={last['test_acc']:.4f}"
                )


if __name__ == "__main__":
    main()

