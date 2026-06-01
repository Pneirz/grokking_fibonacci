# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `iter_lstm_raw_8` | 0.996816 [0.996815, 0.996817] | 0.999844 [0.999843, 0.999844] | 0.999999 [0.999999, 0.999999] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `iter_lstm_raw_8` | 378 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `iter_lstm_raw_8` | 0 |  |  |  | 0.376156 | 0.999999 |
| `iter_lstm_raw_8` | 1 |  |  |  | 0.380308 | 0.999999 |
| `iter_lstm_raw_8` | 2 |  |  |  | 0.378533 | 0.999999 |
| `iter_lstm_raw_8` | 3 |  |  |  | 0.397572 | 0.999999 |
| `iter_lstm_raw_8` | 4 |  |  |  | 0.378107 | 0.999999 |
| `iter_lstm_raw_8` | 5 |  |  |  | 0.382663 | 0.999999 |
| `iter_lstm_raw_8` | 6 |  |  |  | 0.377224 | 0.999999 |
| `iter_lstm_raw_8` | 7 |  |  |  | 0.378631 | 0.999999 |
| `iter_lstm_raw_8` | 8 |  |  |  | 0.386743 | 0.999999 |
| `iter_lstm_raw_8` | 9 |  |  |  | 0.385182 | 0.999999 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `iter_lstm_raw_8` | 0 | 1000000 | 0.999999 | 0.376156 | optimization_failure |
| `iter_lstm_raw_8` | 1 | 1000000 | 0.999999 | 0.380308 | optimization_failure |
| `iter_lstm_raw_8` | 2 | 1000000 | 0.999999 | 0.378533 | optimization_failure |
| `iter_lstm_raw_8` | 3 | 1000000 | 0.999999 | 0.397572 | optimization_failure |
| `iter_lstm_raw_8` | 4 | 1000000 | 0.999999 | 0.378107 | optimization_failure |
| `iter_lstm_raw_8` | 5 | 1000000 | 0.999999 | 0.382663 | optimization_failure |
| `iter_lstm_raw_8` | 6 | 1000000 | 0.999999 | 0.377224 | optimization_failure |
| `iter_lstm_raw_8` | 7 | 1000000 | 0.999999 | 0.378631 | optimization_failure |
| `iter_lstm_raw_8` | 8 | 1000000 | 0.999999 | 0.386743 | optimization_failure |
| `iter_lstm_raw_8` | 9 | 1000000 | 0.999999 | 0.385182 | optimization_failure |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
