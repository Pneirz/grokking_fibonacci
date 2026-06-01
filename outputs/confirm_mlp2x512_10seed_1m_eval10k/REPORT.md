# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `mlp_raw_2x512` | 0.462919 [0.441015, 0.4754] | 0.944591 [0.938664, 0.9469] | 0.999438 [0.999351, 0.999464] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `mlp_raw_2x512` | 264193 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_raw_2x512` | 0 | 630000 |  |  | 0.00824862 | 0.999377 |
| `mlp_raw_2x512` | 1 | 160000 |  |  | 0.00216759 | 0.999466 |
| `mlp_raw_2x512` | 2 |  |  |  | 0.293032 | 0.999342 |
| `mlp_raw_2x512` | 3 |  |  |  | 0.0170153 | 0.999284 |
| `mlp_raw_2x512` | 4 | 90000 |  |  | 0.00151079 | 0.999465 |
| `mlp_raw_2x512` | 5 |  |  |  | 0.019765 | 0.999427 |
| `mlp_raw_2x512` | 6 | 280000 |  |  | 0.0189806 | 0.999459 |
| `mlp_raw_2x512` | 7 | 150000 |  |  | 0.000541859 | 0.999465 |
| `mlp_raw_2x512` | 8 |  |  |  | 1.7515 | 0.999211 |
| `mlp_raw_2x512` | 9 |  |  |  | 0.025317 | 0.999449 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `mlp_raw_2x512` | 0 | 50000 | 0.999101 | 0.310346 | post_fit_no_far_success |
| `mlp_raw_2x512` | 1 | 20000 | 0.999425 | 0.0683236 | post_fit_no_far_success |
| `mlp_raw_2x512` | 2 | 50000 | 0.999297 | 0.636073 | optimization_failure |
| `mlp_raw_2x512` | 3 | 190000 | 0.999205 | 5.33577 | optimization_failure |
| `mlp_raw_2x512` | 4 | 20000 | 0.999453 | 0.0564379 | post_fit_no_far_success |
| `mlp_raw_2x512` | 5 | 50000 | 0.9994 | 0.0859411 | optimization_failure |
| `mlp_raw_2x512` | 6 | 130000 | 0.999427 | 0.0337063 | unstable_train_fit |
| `mlp_raw_2x512` | 7 | 20000 | 0.999418 | 0.0561273 | post_fit_no_far_success |
| `mlp_raw_2x512` | 8 | 370000 | 0.999204 | 8.44551 | optimization_failure |
| `mlp_raw_2x512` | 9 | 100000 | 0.999424 | 0.212118 | optimization_failure |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
