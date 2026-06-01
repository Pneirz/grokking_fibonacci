# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `nac_raw` | 0.953441 [0.953441, 0.953441] | 0.997326 [0.997326, 0.997326] | 0.999981 [0.999981, 0.999981] |
| `nalu_raw` | 0.110986 [0.0849332, 0.141166] | 0.636922 [0.597559, 0.680571] | 0.965906 [0.964139, 0.969823] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `nac_raw` | 8448 |
| `nalu_raw` | 12801 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `nac_raw` | 0 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 1 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 2 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 3 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 4 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 5 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 6 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 7 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 8 |  |  |  | 107.117 | 0.999981 |
| `nac_raw` | 9 |  |  |  | 107.117 | 0.999981 |
| `nalu_raw` | 0 | 740000 |  |  | 0.0215293 | 0.947253 |
| `nalu_raw` | 1 |  |  |  | 0.0163735 | 0.96585 |
| `nalu_raw` | 2 |  |  |  | 0.0376824 | 0.970523 |
| `nalu_raw` | 3 | 510000 |  |  | 0.00694609 | 1 |
| `nalu_raw` | 4 |  |  |  | 0.0364816 | 0.965962 |
| `nalu_raw` | 5 | 940000 |  |  | 0.037862 | 0.967724 |
| `nalu_raw` | 6 | 750000 |  |  | 0.0228767 | 0.965497 |
| `nalu_raw` | 7 | 950000 |  |  | 0.0344901 | 0.973056 |
| `nalu_raw` | 8 | 870000 |  |  | 0.0157198 | 0.963686 |
| `nalu_raw` | 9 |  |  |  | 0.0119367 | 0.95946 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `nac_raw` | 0 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 1 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 2 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 3 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 4 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 5 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 6 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 7 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 8 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nac_raw` | 9 | 40000 | 0.999981 | 107.117 | optimization_failure |
| `nalu_raw` | 0 | 30000 | 0.907237 | 0.222899 | unstable_train_fit |
| `nalu_raw` | 1 | 20000 | 0.919935 | 0.300143 | optimization_failure |
| `nalu_raw` | 2 | 50000 | 0.883074 | 0.192288 | optimization_failure |
| `nalu_raw` | 3 | 20000 | 0.913693 | 0.289392 | post_fit_no_far_success |
| `nalu_raw` | 4 | 80000 | 0.965288 | 0.311889 | optimization_failure |
| `nalu_raw` | 5 | 20000 | 0.885647 | 0.164018 | unstable_train_fit |
| `nalu_raw` | 6 | 10000 | 0.891277 | 0.118493 | unstable_train_fit |
| `nalu_raw` | 7 | 80000 | 0.941952 | 0.275409 | unstable_train_fit |
| `nalu_raw` | 8 | 30000 | 0.937408 | 0.278522 | unstable_train_fit |
| `nalu_raw` | 9 | 30000 | 0.893996 | 0.0995009 | optimization_failure |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
