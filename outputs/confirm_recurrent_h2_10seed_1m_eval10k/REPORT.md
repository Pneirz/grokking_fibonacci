# Index-Only Fibonacci Extrapolation Report

Input feature: raw scalar `n` only.

Train range: `n=10..30`.
Near: `31..35`; mid: `36..45`; far: `46..60`.

Primary neural targets are raw Fibonacci values: `F_n`. Any scaled or log-space model is an optional diagnostic, not part of the raw primary comparison.

## Median Mean Relative Absolute Error

| Model | Near | Mid | Far |
|---|---:|---:|---:|
| `linear_recurrence_raw_2` | 3.64321e-07 [1.78407e-07, 4.10471e-07] | 1.34061e-06 [5.71267e-07, 1.55055e-06] | 3.01947e-06 [1.23084e-06, 3.52508e-06] |

## Parameter Counts

| Model | Trainable parameters |
|---|---:|
| `linear_recurrence_raw_2` | 11 |

## Learning Events

Train fit threshold: `0.01` mean relative error.
Far extrapolation threshold: `0.05` mean relative error.

| Model | Seed | First train fit | First far extrapolation | Gap | Final train | Final far |
|---|---:|---:|---:|---:|---:|---:|
| `linear_recurrence_raw_2` | 0 | 20000 | 20000 | 0 | 3.85531e-05 | 1.81481e-06 |
| `linear_recurrence_raw_2` | 1 | 20000 | 20000 | 0 | 1.1614e-05 | 9.13138e-07 |
| `linear_recurrence_raw_2` | 2 | 20000 | 20000 | 0 | 7.27742e-05 | 3.49243e-06 |
| `linear_recurrence_raw_2` | 3 | 20000 | 20000 | 0 | 3.81062e-05 | 2.63056e-06 |
| `linear_recurrence_raw_2` | 4 | 20000 | 10000 | -10000 | 7.34104e-05 | 3.53596e-06 |
| `linear_recurrence_raw_2` | 5 | 20000 | 20000 | 0 | 0.000207656 | 8.68819e-06 |
| `linear_recurrence_raw_2` | 6 | 10000 | 10000 | 0 | 1.08902e-05 | 4.54867e-07 |
| `linear_recurrence_raw_2` | 7 | 20000 | 20000 | 0 | 5.58402e-05 | 3.40838e-06 |
| `linear_recurrence_raw_2` | 8 | 20000 | 20000 | 0 | 5.13792e-05 | 3.78996e-06 |
| `linear_recurrence_raw_2` | 9 | 20000 | 20000 | 0 | 2.33384e-05 | 1.03618e-06 |

## Best Far Checkpoints

| Model | Seed | Best far step | Best far | Train at best far | Status |
|---|---:|---:|---:|---:|---|
| `linear_recurrence_raw_2` | 0 | 530000 | 4.10704e-07 | 0.000187469 | successful_not_delayed |
| `linear_recurrence_raw_2` | 1 | 970000 | 6.16682e-07 | 1.29982e-05 | successful_not_delayed |
| `linear_recurrence_raw_2` | 2 | 240000 | 1.36737e-06 | 0.00031096 | successful_not_delayed |
| `linear_recurrence_raw_2` | 3 | 610000 | 5.6181e-07 | 7.95121e-05 | successful_not_delayed |
| `linear_recurrence_raw_2` | 4 | 920000 | 1.46984e-06 | 7.16042e-05 | successful_not_delayed |
| `linear_recurrence_raw_2` | 5 | 460000 | 1.21022e-06 | 0.000177542 | successful_not_delayed |
| `linear_recurrence_raw_2` | 6 | 1000000 | 4.54867e-07 | 1.08902e-05 | successful_not_delayed |
| `linear_recurrence_raw_2` | 7 | 810000 | 2.71489e-06 | 4.20422e-05 | successful_not_delayed |
| `linear_recurrence_raw_2` | 8 | 980000 | 3.81653e-07 | 5.40871e-05 | successful_not_delayed |
| `linear_recurrence_raw_2` | 9 | 950000 | 6.21325e-07 | 2.5794e-05 | successful_not_delayed |

## Reading The Table

The log-linear baseline is the adversarial ceiling: a model that cannot beat or approach it has not discovered the asymptotic exponential law (include `log_linear` explicitly when this reference is needed). The polynomial baseline is the prefix-fit trap: low train error with bad future error is evidence against interpreting train fit as grokking.

A run is treated as meaningful no-grokking evidence only after the train-fit threshold is reached. Runs without train fit are labeled `optimization_failure` rather than no-grokking evidence.
