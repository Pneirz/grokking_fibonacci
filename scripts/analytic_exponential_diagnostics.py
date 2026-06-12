"""Analytic exponential diagnostics for the short raw-index protocol.

Fits two closed-form exponential baselines on the observed training segment
and reports mean relative absolute error per band, using the same split as
the index-only study (train n=10..30; near 31..35; mid 36..45; far 46..60):

- ``c r^n``: least-squares fit in log space, i.e. ordinary least squares of
  log(F(n)) on n. The fitted growth factor is r = 1.6180350, within about
  1e-6 of the golden ratio.
- ``c r^n + d``: nonlinear least squares on raw values (scipy curve_fit).
  This fit essentially recovers Binet's asymptotic law (c close to 1/sqrt(5)
  and r close to phi); mid/far errors land in the 1e-9 to 1e-8 regime, with
  exact values depending on optimizer tolerance.

Like the log-linear baseline, these diagnostics show that the split contains
exploitable exponential structure: relative far-band success is reachable by
tracking the dominant exponential trend alone, without recovering the exact
integer recurrence.
"""

import numpy as np
from scipy.optimize import curve_fit


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def band_rel_error(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[mask] - true[mask]) / np.maximum(true[mask], 1.0)))


def main() -> None:
    ns = np.arange(10, 61)
    fib = np.array([fibonacci(int(n)) for n in ns], dtype=np.float64)
    bands = {
        "train": (ns >= 10) & (ns <= 30),
        "near": (ns >= 31) & (ns <= 35),
        "mid": (ns >= 36) & (ns <= 45),
        "far": (ns >= 46) & (ns <= 60),
    }
    x_train = ns[bands["train"]].astype(np.float64)
    y_train = fib[bands["train"]]

    # c r^n via least squares in log space.
    slope, intercept = np.polyfit(x_train, np.log(y_train), deg=1)
    r_log, c_log = float(np.exp(slope)), float(np.exp(intercept))
    pred_log = c_log * r_log ** ns

    # c r^n + d via nonlinear least squares on raw values.
    params, _ = curve_fit(
        lambda n, c, r, d: c * r ** n + d, x_train, y_train, p0=[1.0, 1.6, 0.0], maxfev=20000
    )
    c_off, r_off, d_off = (float(v) for v in params)
    pred_off = c_off * r_off ** ns + d_off

    phi = (1.0 + 5.0 ** 0.5) / 2.0
    print(f"c r^n     (log-space LS): c={c_log:.8f} r={r_log:.10f}  (phi={phi:.10f})")
    print(f"c r^n + d (raw LS):       c={c_off:.8f} r={r_off:.10f} d={d_off:.6f}")
    header = f"{'model':14s}" + "".join(f"{b:>12s}" for b in bands)
    print(header)
    for name, pred in [("exp", pred_log), ("exp_offset", pred_off)]:
        row = f"{name:14s}" + "".join(
            f"{band_rel_error(pred, fib, mask):12.3e}" for mask in bands.values()
        )
        print(row)


if __name__ == "__main__":
    main()
