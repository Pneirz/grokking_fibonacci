# Grokking Fibonacci Experiments

This repository contains reproducible experiment code and archived outputs for
index-only Fibonacci extrapolation studies.

The core task is deliberately strict: models receive only the scalar index `n`
and are trained to predict the raw Fibonacci value `F_n`. The archived results
compare direct feedforward models, generic feature controls, arithmetic-bias
baselines, generic recurrent controls, and a small recurrent-state positive
control.

## Contents

- `fibonacci_grokking/`: Python package with data generation, models, training,
  analysis, and raw-index extrapolation code.
- `scripts/plot_short_dynamics.py`: helper for plotting training dynamics from
  saved curve CSV files.
- `outputs/confirm_*`: archived ten-seed, one-million-update raw-index runs.
- `outputs/short_raw_analytic_baselines`: log-linear and polynomial diagnostic
  baselines for the short raw-index split.
- `outputs/modular_*.csv`: archived modular arithmetic control runs.
- `PROTOCOL.md`: task definition, metrics, model roles, and evaluation rules.
- `RUNS.md`: commands corresponding to the archived experiment families.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency list includes PyTorch, plotting dependencies, PyKAN, and
`efficient-kan`.

## Quick Smoke Test

```powershell
.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models poly_raw,linear_raw --seeds 0 --steps 200 --eval-every 50 --output-dir outputs\smoke_index_study
```

The command writes `results.csv`, `curves.csv`, `predictions.csv`, and
`REPORT.md` into the requested output directory.
