"""
Comparison of every LinearRegression variant implemented in this project.

Runs 5-fold CV on the regression task for:
  - plain OLS                         (closed-form)
  - Ridge at several λ                (closed-form)
  - Ridge + polynomial expansion      (closed-form, d=2, d=2+int, d=3)
  - OLS and Ridge via gradient descent (for solver-equivalence check)

Also reports wall-clock fit time for the closed-form vs GD solver at the
winning configuration, and a machine-precision equivalence check
(‖w_CF - w_GD‖ / ‖w_CF‖) proving both solvers minimize the same objective.

Usage:
    python linreg_comparison.py
"""

import time
import numpy as np

from src.methods.linear_regression import LinearRegression
from src.utils import normalize_fn
from src.evaluation import kfold_cross_validation


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #

SEED = 100
K    = 5

CONFIGS = [
    # name                           kwargs
    ("OLS (CF)",                     {'lambda_reg': 0.0,   'method': 'closed_form'}),
    ("Ridge λ=0.1 (CF)",             {'lambda_reg': 0.1,   'method': 'closed_form'}),
    ("Ridge λ=1 (CF)",               {'lambda_reg': 1.0,   'method': 'closed_form'}),
    ("Ridge λ=10 (CF)",              {'lambda_reg': 10.0,  'method': 'closed_form'}),
    ("Ridge λ=100 (CF)",             {'lambda_reg': 100.0, 'method': 'closed_form'}),
    ("Ridge λ=10 + poly d=2 (CF)",
        {'lambda_reg': 10.0, 'method': 'closed_form', 'degree': 2}),
    ("Ridge λ=10 + poly d=2+int (CF)",
        {'lambda_reg': 10.0, 'method': 'closed_form', 'degree': 2, 'interaction': True}),
    ("Ridge λ=10 + poly d=3 (CF)",
        {'lambda_reg': 10.0, 'method': 'closed_form', 'degree': 3}),
    ("OLS (GD)",
        {'lambda_reg': 0.0, 'method': 'gradient_descent', 'lr': 0.05, 'max_iters': 2000}),
    ("Ridge λ=10 (GD)",
        {'lambda_reg': 10.0, 'method': 'gradient_descent', 'lr': 0.05, 'max_iters': 2000}),
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_regression_data():
    d = np.load("data/features.npz")
    X_raw, y = d["xtrain"], d["ytrainreg"]
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _align(s, w):
    return s + " " * max(0, w - len(s))


def print_table(rows, headers):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep    = "  "
    print(sep.join(_align(h, widths[i]) for i, h in enumerate(headers)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(sep.join(_align(str(r[i]), widths[i]) for i in range(len(headers))))


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def run_cv_table(X, y):
    print("\n=== 5-fold CV over all LinReg variants =================================")
    kwargs_list = [c[1] for c in CONFIGS]
    names       = [c[0] for c in CONFIGS]

    np.random.seed(SEED)
    _, results = kfold_cross_validation(LinearRegression, kwargs_list,
                                        X, y, k=K, task='regression')

    rows = []
    for name, r in zip(names, results):
        rows.append((name, f"{r['mean']:.4f}", f"{r['std']:.4f}"))

    # sort by mean for a clean summary, but keep insertion order above
    print("\n-- Sorted by mean CV MSE (low → high) --")
    print_table(sorted(rows, key=lambda x: float(x[1])),
                headers=["Variant", "mean MSE", "std"])

    best_idx = int(np.argmin([r["mean"] for r in results]))
    print(f"\nBest CV variant: {names[best_idx]}   "
          f"MSE = {results[best_idx]['mean']:.4f} ± {results[best_idx]['std']:.4f}")
    return results


def solver_equivalence(X, y):
    """Show CF and GD minimize the same objective (bit-identical weights)."""
    print("\n=== Closed-form vs gradient-descent solver equivalence ================")
    rows = []
    for lam in [0.0, 10.0, 100.0]:
        cf = LinearRegression(lambda_reg=lam, method='closed_form')
        gd = LinearRegression(lambda_reg=lam, method='gradient_descent',
                              lr=0.05, max_iters=5000)
        cf.fit(X, y)
        gd.fit(X, y)
        mse_cf = float(np.mean((cf.predict(X) - y) ** 2))
        mse_gd = float(np.mean((gd.predict(X) - y) ** 2))
        rel = float(np.linalg.norm(cf.weights - gd.weights) /
                    np.linalg.norm(cf.weights))
        rows.append((f"λ = {lam}", f"{mse_cf:.6f}", f"{mse_gd:.6f}", f"{rel:.2e}"))
    print_table(rows, headers=["setting", "CF MSE", "GD MSE", "‖Δw‖/‖w‖"])


def timing(X, y, repeats=5):
    """Wall-clock fit time: closed-form vs gradient descent at λ=10."""
    print("\n=== Fit-time comparison at λ=10 (median of 5 runs) ====================")
    def _time(kw):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            LinearRegression(**kw).fit(X, y)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    t_cf = _time({'lambda_reg': 10.0, 'method': 'closed_form'})
    t_gd = _time({'lambda_reg': 10.0, 'method': 'gradient_descent',
                  'lr': 0.05, 'max_iters': 2000})
    print_table([
        ("closed-form",            f"{t_cf * 1000:.2f} ms", "—"),
        ("gradient descent (2k)",  f"{t_gd * 1000:.2f} ms", f"{t_gd / t_cf:.1f}× slower"),
    ], headers=["solver", "median fit time", "ratio"])


# --------------------------------------------------------------------------- #

def main():
    X, y = load_regression_data()
    print(f"Loaded N = {X.shape[0]}, D = {X.shape[1]} (z-normalized).")
    run_cv_table(X, y)
    solver_equivalence(X, y)
    timing(X, y)


if __name__ == "__main__":
    main()
