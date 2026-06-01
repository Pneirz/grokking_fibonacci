# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `multkan_raw_medium` | 0.651788 [0.557016, 0.722998] | 0.99511 [0.992213, 0.999542] | 0.999966 [0.999934, 1.00001] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `multkan_raw_medium` | 442 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `multkan_raw_medium` | 0 | 20000 |  |  | 0.000107656 | 0.999922 |
| `multkan_raw_medium` | 1 | 30000 |  |  | 2.96025e-08 | 0.999932 |
| `multkan_raw_medium` | 2 | 80000 |  |  | 6.37578e-07 | 1 |
| `multkan_raw_medium` | 3 | 30000 |  |  | 0.000448053 | 0.999944 |
| `multkan_raw_medium` | 4 | 50000 |  |  | 0.000237437 | 1.00001 |
| `multkan_raw_medium` | 5 | 90000 |  |  | 0.00190188 | 1.00001 |
| `multkan_raw_medium` | 6 | 30000 |  |  | 2.862e-05 | 0.999932 |
| `multkan_raw_medium` | 7 | 50000 |  |  | 0.0128257 | 1.00001 |
| `multkan_raw_medium` | 8 | 60000 |  |  | 8.0337e-05 | 0.999987 |
| `multkan_raw_medium` | 9 | 30000 |  |  | 0.000104065 | 0.999941 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `multkan_raw_medium` | 0 | 150000 | 0.999922 | 0.000312127 | post_fit_no_far_success |
| `multkan_raw_medium` | 1 | 30000 | 0.999932 | 0.000133073 | post_fit_no_far_success |
| `multkan_raw_medium` | 2 | 50000 | 0.999968 | 0.107408 | post_fit_no_far_success |
| `multkan_raw_medium` | 3 | 20000 | 0.999943 | 0.0272577 | post_fit_no_far_success |
| `multkan_raw_medium` | 4 | 30000 | 0.99998 | 0.0658451 | post_fit_no_far_success |
| `multkan_raw_medium` | 5 | 60000 | 0.99999 | 0.456169 | post_fit_no_far_success |
| `multkan_raw_medium` | 6 | 1000000 | 0.999932 | 2.862e-05 | post_fit_no_far_success |
| `multkan_raw_medium` | 7 | 30000 | 0.99998 | 0.0554063 | unstable_train_fit |
| `multkan_raw_medium` | 8 | 40000 | 0.999976 | 0.0337237 | post_fit_no_far_success |
| `multkan_raw_medium` | 9 | 20000 | 0.999939 | 0.0900857 | post_fit_no_far_success |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
