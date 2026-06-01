# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `mul_mlp_raw_2x256` | 0.499965 [0.479856, 0.518002] | 0.935498 [0.930401, 0.940073] | 0.99883 [0.998702, 0.998932] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `mul_mlp_raw_2x256` | 263937 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `mul_mlp_raw_2x256` | 0 | 70000 |  |  | 0.00209307 | 0.998653 |
| `mul_mlp_raw_2x256` | 1 | 70000 |  |  | 0.638736 | 0.998854 |
| `mul_mlp_raw_2x256` | 2 | 620000 |  |  | 0.00665772 | 0.9987 |
| `mul_mlp_raw_2x256` | 3 |  |  |  | 0.0150963 | 0.998708 |
| `mul_mlp_raw_2x256` | 4 | 640000 |  |  | 0.0021879 | 0.998934 |
| `mul_mlp_raw_2x256` | 5 | 130000 |  |  | 0.0033816 | 0.998943 |
| `mul_mlp_raw_2x256` | 6 | 210000 |  |  | 0.0168355 | 0.998966 |
| `mul_mlp_raw_2x256` | 7 | 280000 |  |  | 0.00591503 | 0.998807 |
| `mul_mlp_raw_2x256` | 8 | 90000 |  |  | 0.00712794 | 0.998616 |
| `mul_mlp_raw_2x256` | 9 | 70000 |  |  | 0.0010641 | 0.998926 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `mul_mlp_raw_2x256` | 0 | 580000 | 0.998629 | 0.255398 | post_fit_no_far_success |
| `mul_mlp_raw_2x256` | 1 | 280000 | 0.998815 | 0.00100003 | unstable_train_fit |
| `mul_mlp_raw_2x256` | 2 | 730000 | 0.998679 | 0.662531 | post_fit_no_far_success |
| `mul_mlp_raw_2x256` | 3 | 630000 | 0.998696 | 0.0285045 | optimization_failure |
| `mul_mlp_raw_2x256` | 4 | 570000 | 0.998922 | 0.349331 | post_fit_no_far_success |
| `mul_mlp_raw_2x256` | 5 | 850000 | 0.998937 | 0.192138 | post_fit_no_far_success |
| `mul_mlp_raw_2x256` | 6 | 370000 | 0.998957 | 0.110413 | unstable_train_fit |
| `mul_mlp_raw_2x256` | 7 | 110000 | 0.998782 | 0.04271 | post_fit_no_far_success |
| `mul_mlp_raw_2x256` | 8 | 630000 | 0.99858 | 0.191198 | post_fit_no_far_success |
| `mul_mlp_raw_2x256` | 9 | 780000 | 0.998922 | 0.00186402 | post_fit_no_far_success |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
