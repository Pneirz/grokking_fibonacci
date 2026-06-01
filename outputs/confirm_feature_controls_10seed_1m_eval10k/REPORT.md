# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `fourier_mlp_raw` | 0.953538 [0.952511, 0.954658] | 0.992475 [0.99241, 0.992526] | 0.999946 [0.999946, 0.999946] |
| `siren_raw` | 0.994826 [0.98964, 0.999516] | 0.99978 [0.999532, 0.999976] | 0.999999 [0.999997, 1] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `fourier_mlp_raw` | 37505 |
| `siren_raw` | 33409 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `fourier_mlp_raw` | 0 |  |  |  | 0.43559 | 0.999946 |
| `fourier_mlp_raw` | 1 |  |  |  | 0.427528 | 0.999948 |
| `fourier_mlp_raw` | 2 |  |  |  | 0.426563 | 0.999946 |
| `fourier_mlp_raw` | 3 |  |  |  | 0.0610501 | 0.999946 |
| `fourier_mlp_raw` | 4 |  |  |  | 0.424961 | 0.999946 |
| `fourier_mlp_raw` | 5 |  |  |  | 0.427559 | 0.999946 |
| `fourier_mlp_raw` | 6 |  |  |  | 0.0307565 | 0.999946 |
| `fourier_mlp_raw` | 7 |  |  |  | 0.056863 | 0.999946 |
| `fourier_mlp_raw` | 8 |  |  |  | 0.524813 | 0.999946 |
| `fourier_mlp_raw` | 9 |  |  |  | 0.426454 | 0.999947 |
| `siren_raw` | 0 |  |  |  | 12.837 | 0.999999 |
| `siren_raw` | 1 |  |  |  | 47.8319 | 0.999998 |
| `siren_raw` | 2 |  |  |  | 5.86918 | 1 |
| `siren_raw` | 3 |  |  |  | 80.8628 | 0.999997 |
| `siren_raw` | 4 |  |  |  | 2.35342 | 1 |
| `siren_raw` | 5 |  |  |  | 78.5526 | 0.999997 |
| `siren_raw` | 6 |  |  |  | 2.35232 | 1 |
| `siren_raw` | 7 |  |  |  | 55.069 | 0.999998 |
| `siren_raw` | 8 |  |  |  | 2.33717 | 1 |
| `siren_raw` | 9 |  |  |  | 59.8035 | 0.999997 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `fourier_mlp_raw` | 0 | 10000 | 0.999938 | 0.37758 | optimization_failure |
| `fourier_mlp_raw` | 1 | 10000 | 0.99994 | 0.33058 | optimization_failure |
| `fourier_mlp_raw` | 2 | 10000 | 0.999939 | 0.284082 | optimization_failure |
| `fourier_mlp_raw` | 3 | 10000 | 0.999941 | 0.239854 | optimization_failure |
| `fourier_mlp_raw` | 4 | 10000 | 0.99994 | 0.330499 | optimization_failure |
| `fourier_mlp_raw` | 5 | 10000 | 0.99994 | 0.377932 | optimization_failure |
| `fourier_mlp_raw` | 6 | 10000 | 0.999939 | 0.377996 | optimization_failure |
| `fourier_mlp_raw` | 7 | 10000 | 0.999939 | 0.377773 | optimization_failure |
| `fourier_mlp_raw` | 8 | 10000 | 0.999939 | 0.377738 | optimization_failure |
| `fourier_mlp_raw` | 9 | 10000 | 0.999937 | 0.377655 | optimization_failure |
| `siren_raw` | 0 | 1000000 | 0.999999 | 12.837 | optimization_failure |
| `siren_raw` | 1 | 970000 | 0.999998 | 55.5301 | optimization_failure |
| `siren_raw` | 2 | 1000000 | 1 | 5.86918 | optimization_failure |
| `siren_raw` | 3 | 950000 | 0.999997 | 73.4053 | optimization_failure |
| `siren_raw` | 4 | 1000000 | 1 | 2.35342 | optimization_failure |
| `siren_raw` | 5 | 990000 | 0.999997 | 73.5318 | optimization_failure |
| `siren_raw` | 6 | 1000000 | 1 | 2.35232 | optimization_failure |
| `siren_raw` | 7 | 990000 | 0.999998 | 51.0211 | optimization_failure |
| `siren_raw` | 8 | 1000000 | 1 | 2.33717 | optimization_failure |
| `siren_raw` | 9 | 990000 | 0.999997 | 74.3351 | optimization_failure |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
