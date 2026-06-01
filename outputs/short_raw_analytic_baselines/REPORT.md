# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `log_linear` | 0.0044247 [0.0044247, 0.0044247] | 0.00825641 [0.00825641, 0.00825641] | 0.0146104 [0.0146104, 0.0146104] |
| `poly_raw` | 0.0249368 [0.0249368, 0.0249368] | 23.2966 [23.2966, 23.2966] | 652.868 [652.868, 652.868] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `log_linear` | analytic |
| `poly_raw` | analytic |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.
