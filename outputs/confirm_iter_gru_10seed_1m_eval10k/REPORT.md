# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `iter_gru_raw_8` | 0.996823 [0.996815, 0.996828] | 0.999844 [0.999843, 0.999844] | 0.999999 [0.999999, 0.999999] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `iter_gru_raw_8` | 282 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `iter_gru_raw_8` | 0 |  |  |  | 0.396142 | 0.999999 |
| `iter_gru_raw_8` | 1 |  |  |  | 0.409536 | 0.999999 |
| `iter_gru_raw_8` | 2 |  |  |  | 0.378621 | 0.999999 |
| `iter_gru_raw_8` | 3 |  |  |  | 0.388738 | 0.999999 |
| `iter_gru_raw_8` | 4 |  |  |  | 0.375724 | 0.999999 |
| `iter_gru_raw_8` | 5 |  |  |  | 0.396529 | 0.999999 |
| `iter_gru_raw_8` | 6 |  |  |  | 0.380498 | 0.999999 |
| `iter_gru_raw_8` | 7 |  |  |  | 0.408069 | 0.999999 |
| `iter_gru_raw_8` | 8 |  |  |  | 234.375 | 0.99999 |
| `iter_gru_raw_8` | 9 |  |  |  | 234.374 | 0.99999 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `iter_gru_raw_8` | 0 | 1000000 | 0.999999 | 0.396142 | optimization_failure |
| `iter_gru_raw_8` | 1 | 1000000 | 0.999999 | 0.409536 | optimization_failure |
| `iter_gru_raw_8` | 2 | 1000000 | 0.999999 | 0.378621 | optimization_failure |
| `iter_gru_raw_8` | 3 | 1000000 | 0.999999 | 0.388738 | optimization_failure |
| `iter_gru_raw_8` | 4 | 1000000 | 0.999999 | 0.375724 | optimization_failure |
| `iter_gru_raw_8` | 5 | 1000000 | 0.999999 | 0.396529 | optimization_failure |
| `iter_gru_raw_8` | 6 | 1000000 | 0.999999 | 0.380498 | optimization_failure |
| `iter_gru_raw_8` | 7 | 1000000 | 0.999999 | 0.408069 | optimization_failure |
| `iter_gru_raw_8` | 8 | 1000000 | 0.99999 | 234.375 | optimization_failure |
| `iter_gru_raw_8` | 9 | 1000000 | 0.99999 | 234.374 | optimization_failure |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
