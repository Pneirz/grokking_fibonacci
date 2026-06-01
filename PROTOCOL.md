# Index-Only Fibonacci Extrapolation Protocol

## Core Claim

The experiment asks a narrow question:

> Given only the scalar index `n`, can a learned model extrapolate raw Fibonacci
> values outside the training prefix?

No model receives `(F_n, F_{n+1})`, recurrence state, modular features, or any
engineered Fibonacci feature. The only input is the raw scalar index:

```text
x = n
```

The primary output and the training target are the raw Fibonacci value:

```text
y = F_n
```

## Strict Primary Rule

The primary claim is intentionally strict. We are trying to get grokking without
adding mathematical shortcuts.

Allowed:

- More optimization steps.
- Larger or deeper generic networks.
- Different generic model classes such as MLP, PyKAN, or efficient-KAN.
- Generic optimizer changes such as learning-rate schedules or gradient
  clipping.

Disallowed for the primary claim:

- Input normalization such as `n / n_train`.
- Target scaling such as `F_n / F_train`.
- Log targets such as `log(F_n)`.
- Recurrence-state inputs such as `(F_n, F_{n+1})`.
- Hand-coded exponential, Binet, recurrence-derived, or Fibonacci-specific
  features. Generic functional bases such as random Fourier features are
  reported separately as feature controls, not as evidence that the strict
  raw-scalar interface itself is sufficient.

## Literature Context

- Grokking was framed as delayed generalization after overfitting on small
  algorithmic datasets: https://arxiv.org/abs/2201.02177
- KANs replace fixed node activations with learned edge-wise spline functions
  and are proposed as interpretable alternatives to MLPs:
  https://arxiv.org/abs/2404.19756
- PyKAN examples commonly use small KANs such as `width=[2,5,1]`, `grid=5`,
  `k=3`, and LBFGS-style short training:
  https://kindxiaoming.github.io/pykan/API_demo/API_6_training_hyperparameter.html
- `efficient-kan` reformulates spline evaluation as matrix multiplication to
  reduce memory cost, and replaces sample-based L1 regularization with
  weight-based L1 regularization compatible with that reformulation:
  https://github.com/Blealtan/efficient-kan

## Why This Is Interesting

This is an adversarial extrapolation benchmark. The training prefix is tiny and
perfectly deterministic, so many flexible models can fit it. The real question
is whether their learned function has the right out-of-domain growth law.

The study is deliberately simple:

- One input variable: `n`.
- One sequence: Fibonacci.
- One split: prefix train, future extrapolation.
- Multiple horizons: near, mid, far.
- Strong classical baselines: a high-degree raw polynomial trap and an optional
  log-linear reference.

Impact angle: flexible neural approximators can look excellent on deterministic
train prefixes while failing raw future magnitudes. KANs are especially
interesting because spline-based flexibility may fit the prefix while
extrapolating poorly outside the training grid.

## Main Dataset

Default:

```text
train: n = 10,...,30
near:  n = 31,...,35
mid:   n = 36,...,45
far:   n = 46,...,60
```

Starting at `n=10` avoids over-weighting the transient early Fibonacci values
where `log(F_n)` is not yet close to the asymptotic Binet line.

## Targets

All metrics are computed on decoded raw Fibonacci values.

Primary training target:

```text
raw = F_n
```

Scaled or log-space models are allowed only as optional diagnostics or
mathematical references. They are not part of the primary raw comparison.

## Models

| Model | Input | Target | Role |
|---|---|---|---|
| `poly_raw` | `n` | `F_n` | Overfit trap: can fit prefix but usually extrapolates badly. |
| `linear_raw` | `n` | `F_n` | Raw affine control. |
| `mlp_raw` | `n` | `F_n` | Flexible direct raw predictor. |
| `mlp_raw_matched_kan` | `n` | `F_n` | MLP with parameter count matched to `kan_raw`. |
| `mlp_raw_matched_kan_medium` | `n` | `F_n` | MLP matched to `kan_raw_medium`. |
| `mlp_raw_matched_kan_wide` | `n` | `F_n` | MLP matched to `kan_raw_wide`. |
| `mlp_raw_wide` | `n` | `F_n` | High-capacity MLP stress test. |
| `mlp_raw_2x512` | `n` | `F_n` | Two-hidden-layer MLP stress test for the raw universal-approximation hypothesis. |
| `kan_raw` | `n` | `F_n` | KAN direct raw predictor. |
| `kan_raw_medium` | `n` | `F_n` | Higher-capacity KAN. |
| `kan_raw_wide` | `n` | `F_n` | Wide KAN stress test. |
| `efficient_kan_raw` | `n` | `F_n` | Efficient KAN backend with the small KAN tier. |
| `mlp_raw_matched_efficient_kan` | `n` | `F_n` | MLP matched to `efficient_kan_raw`. |
| `efficient_kan_raw_medium` | `n` | `F_n` | Efficient KAN medium tier. |
| `mlp_raw_matched_efficient_kan_medium` | `n` | `F_n` | MLP matched to `efficient_kan_raw_medium`. |
| `efficient_kan_raw_wide` | `n` | `F_n` | Efficient KAN wide tier. |
| `mlp_raw_matched_efficient_kan_wide` | `n` | `F_n` | MLP matched to `efficient_kan_raw_wide`. |

Optional reference:

| Model | Input | Target | Role |
|---|---|---|---|
| `log_linear` | `n` | `log1p(F_n)` | Binet-like reference ceiling, not raw training. |

Generic feature controls:

| Model | Input | Target | Role |
|---|---|---|---|
| `fourier_mlp_raw` | generic Fourier basis of `n` | `F_n` | Task-agnostic functional-basis control. It does not encode Fibonacci, Binet, recurrence state, or target transforms, but it relaxes the raw-scalar representation. |
| `siren_raw` | `n` | `F_n` | Sine-activation functional-basis control. |

Arithmetic and recurrent controls:

| Model | Input | Target | Role |
|---|---|---|---|
| `nac_raw` | `n` | `F_n` | NAC arithmetic-bias baseline. |
| `nalu_raw` | `n` | `F_n` | NALU arithmetic-bias baseline. |
| `iter_rnn_raw_8` | iteration count `n` | `F_n` | Generic recurrent-state control with a vanilla RNN cell. |
| `iter_gru_raw_8` | iteration count `n` | `F_n` | Generic recurrent-state control with a GRU cell. |
| `iter_lstm_raw_8` | iteration count `n` | `F_n` | Generic recurrent-state control with an LSTM cell. |
| `linear_recurrence_raw_2` | iteration count `n` | `F_n` | Strong positive control with learned affine state transition. |

Primary neural budget:

```text
MLP raw: 1 -> 128 -> 128 -> 128 -> 1, SiLU, AdamW
MLP matched: one hidden layer with parameter count matched to the paired KAN
KAN: width=[1,5,1], grid=5, k=3, AdamW
```

The KAN grid range is restricted to the training input domain by default. A
grid range that includes the test horizon should be reported separately as a
range-aware diagnostic, not as the primary result.

## Metrics

For each band:

- Mean absolute error.
- Mean relative absolute error:

```text
mean(|prediction - F_n| / max(F_n, 1))
```

- Median relative absolute error.
- Max relative absolute error.
- Mean log absolute error:

```text
mean(|log1p(max(prediction, 0)) - log1p(F_n)|)
```

- Negative prediction count.
- Monotonicity violation count.
- Rounded exact accuracy.

Training diagnostics:

- Train mean relative absolute error.
- Near/mid/far mean relative absolute error curves.
- First step where train relative MAE <= `fit_rel_mae`.
- First step where far-band relative MAE <= `extrap_rel_mae`.
- Gap between these steps.
- Best far checkpoint.
- Final checkpoint.

## Operational Grokking Criterion

For each seed:

- `train-fit`: first evaluated checkpoint with train mean relative absolute
  error at or below `1e-2`.
- `far-success`: first evaluated checkpoint with far mean relative absolute
  error at or below `5e-2`.
- `grokking-like`: train-fit occurs, far-success occurs at a later checkpoint,
  and the final checkpoint still satisfies train-fit.
- `post_fit_no_far_success`: train-fit occurs and remains stable, but
  far-success never occurs.
- `optimization_failure`: train-fit does not occur.

The `1..70` and `1..200` protocols should not be used as central no-grokking
evidence unless a feedforward model reaches clean train fit. Without train fit,
they are optimization stress tests.

## Adversarial Rules

1. No recurrence-state input.
2. No engineered features beyond normalized scalar `n`.
3. No random train/test split; future is strictly out-of-prefix.
4. No hyperparameter selection on the far band. Use the fixed protocol first.
5. Report the polynomial baseline even if it looks bad; it demonstrates that
   prefix fit is not extrapolation.
6. Report parameter counts for KAN and matched MLP pairs.
7. Report all seeds, then median and IQR.

## Reproducible Command

```powershell
.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models poly_raw,linear_raw,kan_raw,mlp_raw_matched_kan,kan_raw_medium,mlp_raw_matched_kan_medium,kan_raw_wide,mlp_raw_matched_kan_wide,efficient_kan_raw,mlp_raw_matched_efficient_kan,efficient_kan_raw_medium,mlp_raw_matched_efficient_kan_medium,efficient_kan_raw_wide,mlp_raw_matched_efficient_kan_wide,mlp_raw_wide --seeds 0,1,2,3,4 --steps 50000 --eval-every 500 --output-dir outputs\index_study_raw
```

Smoke test:

```powershell
.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models poly_raw,linear_raw,efficient_kan_raw,mlp_raw_matched_efficient_kan --seeds 0 --steps 200 --eval-every 50 --output-dir outputs\index_study_raw_smoke
```

## Report Structure

1. Problem statement: index-only raw Fibonacci extrapolation.
2. Adversarial design: why prefix fit is insufficient.
3. Methods: data split, target spaces, models, budgets.
4. Main table: relative error by near/mid/far band.
5. Grokking table: first train fit, first far extrapolation, gap.
6. Failure analysis: polynomial blow-up, spline/KAN range behavior,
   monotonicity/negative predictions.
7. Minimal conclusion: which inductive bias, if any, supports extrapolation.
