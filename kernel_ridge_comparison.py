"""
Comparison of every kernel configuration for:
  - KernelRidgeRegression        (regression task)
  - KernelLeastSquaresClassifier (classification task)

Kernels swept: linear, polynomial (d=2, d=3 at coef0=1), RBF (σ ∈ {1,3,5,10}).
All at λ=1; an additional λ sweep is run at the winning kernel for each task.

For regression we report 5-fold CV MSE; for classification, 5-fold
stratified CV accuracy and macro-F1. We also report a quick "linear
kernel ≡ unbiased linear ridge" check against the separately implemented
LinearRegression.

Usage:
    python kernel_ridge_comparison.py
"""

import time
import numpy as np

from src.methods.extension.kernel_ridge import (
    KernelRidgeRegression, KernelLeastSquaresClassifier,
)
from src.methods.linear_regression import LinearRegression
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn


SEED = 100
K_CV = 5


def load_data():
    d = np.load("data/features.npz")
    X_raw = d["xtrain"]
    y_c   = d["ytrainclassif"].astype(int)
    y_r   = d["ytrainreg"]
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y_c, y_r


def _stratified_folds(y, k, seed):
    rng = np.random.RandomState(seed)
    class_idx = [np.where(y == c)[0].copy() for c in np.unique(y)]
    for ci in class_idx:
        rng.shuffle(ci)
    class_folds = [np.array_split(ci, k) for ci in class_idx]
    return [np.concatenate([cf[i] for cf in class_folds]) for i in range(k)]


def _random_folds(N, k, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(N); rng.shuffle(idx)
    return np.array_split(idx, k)


def cv_regression(X, y, kwargs):
    folds = _random_folds(len(y), K_CV, SEED)
    mses = []
    for i in range(K_CV):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(K_CV) if j != i])
        m = KernelRidgeRegression(**kwargs); m.fit(X[tr], y[tr])
        mses.append(mse_fn(m.predict(X[val]), y[val]))
    return float(np.mean(mses)), float(np.std(mses))


def cv_classification(X, y, kwargs):
    folds = _stratified_folds(y, K_CV, SEED)
    accs, f1s = [], []
    for i in range(K_CV):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(K_CV) if j != i])
        m = KernelLeastSquaresClassifier(**kwargs); m.fit(X[tr], y[tr])
        p = m.predict(X[val])
        accs.append(accuracy_fn(p, y[val])); f1s.append(macrof1_fn(p, y[val]))
    return (float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(f1s)), float(np.std(f1s)))


def median_fit_time(cls, kwargs, X, y, repeats=3):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cls(**kwargs).fit(X, y)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0


def _align(s, w): return s + " " * max(0, w - len(s))


def print_table(rows, headers):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(_align(h, widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(_align(str(r[i]), widths[i]) for i in range(len(headers))))


# --------------------------------------------------------------------------- #

KERNEL_CONFIGS = [
    ("linear",            {'kernel': 'linear'}),
    ("poly d=2 (c=1)",    {'kernel': 'poly', 'degree': 2, 'coef0': 1.0}),
    ("poly d=3 (c=1)",    {'kernel': 'poly', 'degree': 3, 'coef0': 1.0}),
    ("rbf σ=1",           {'kernel': 'rbf', 'sigma': 1.0}),
    ("rbf σ=3",           {'kernel': 'rbf', 'sigma': 3.0}),
    ("rbf σ=5",           {'kernel': 'rbf', 'sigma': 5.0}),
    ("rbf σ=10",          {'kernel': 'rbf', 'sigma': 10.0}),
]


def krr_table(X, yr):
    print("\n=== Kernel Ridge Regression — kernel sweep (5-fold CV, λ=1) ==========")
    rows = []
    for name, kw in KERNEL_CONFIGS:
        mse_m, mse_s = cv_regression(X, yr, {'lambda_reg': 1.0, **kw})
        t_ms = median_fit_time(KernelRidgeRegression, {'lambda_reg': 1.0, **kw}, X, yr)
        rows.append((name, mse_m, mse_s, t_ms))
        print(f"  {name:<20}  MSE = {mse_m:.4f} ± {mse_s:.4f}   time={t_ms:6.0f} ms")
    best = min(rows, key=lambda r: r[1])
    print(f"\nBest KRR kernel: {best[0]}   MSE = {best[1]:.4f}")


def kls_table(X, yc):
    print("\n=== Kernel Least-Squares Classifier — kernel sweep (strat. 5-fold CV, λ=1) ==")
    rows = []
    for name, kw in KERNEL_CONFIGS:
        am, asd, fm, fsd = cv_classification(X, yc, {'lambda_reg': 1.0, **kw})
        t_ms = median_fit_time(KernelLeastSquaresClassifier,
                               {'lambda_reg': 1.0, **kw}, X, yc)
        rows.append((name, am, asd, fm, fsd, t_ms))
        print(f"  {name:<20}  acc={am:5.2f}%  F1={fm:.3f} (±{fsd:.3f})   time={t_ms:6.0f} ms")
    best = max(rows, key=lambda r: r[3])
    print(f"\nBest KLS kernel: {best[0]}   acc={best[1]:.2f}%   F1={best[3]:.3f}")


def linear_equivalence_check(X, yr):
    """Verify the linear-kernel KRR matches un-biased LinReg."""
    print("\n=== KRR linear kernel vs LinearRegression (train MSE) ================")
    # λ=1 both sides, same centering convention
    krr = KernelRidgeRegression(lambda_reg=1.0, kernel='linear')
    krr.fit(X, yr); mse_krr = mse_fn(krr.predict(X), yr)
    lr  = LinearRegression(lambda_reg=1.0, method='closed_form')
    lr.fit(X, yr); mse_lr = mse_fn(lr.predict(X), yr)
    print(f"  KRR linear:             train MSE = {mse_krr:.6f}")
    print(f"  LinearRegression ridge: train MSE = {mse_lr:.6f}")
    print(f"  gap = {abs(mse_krr - mse_lr):.2e}  (expected ≈0 up to numerical precision)")


def rbf_lambda_sweep(X, yr, sigma):
    print(f"\n=== KRR RBF σ={sigma} — λ sweep ================================")
    rows = []
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        mse_m, mse_s = cv_regression(
            X, yr, {'lambda_reg': lam, 'kernel': 'rbf', 'sigma': sigma})
        rows.append((f"λ = {lam}", f"{mse_m:.4f}", f"±{mse_s:.4f}"))
    print_table(rows, headers=["setting", "MSE", "std"])


# --------------------------------------------------------------------------- #

def main():
    X, yc, yr = load_data()
    print(f"Loaded N = {X.shape[0]}, D = {X.shape[1]} (z-normalized).")
    krr_table(X, yr)
    kls_table(X, yc)
    linear_equivalence_check(X, yr)
    rbf_lambda_sweep(X, yr, sigma=5.0)


if __name__ == "__main__":
    main()
