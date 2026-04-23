"""
Comparison of every KNN variant implemented in this project.

For both tasks (classification and regression) we sweep:
  - k  ∈ {1, 3, 5, 7, 10, 15, 17, 20, 30, 50}
  - metric ∈ {l2, l1, cosine}
  - weighted ∈ {False, True}

For classification we report 5-fold stratified CV accuracy and macro-F1;
for regression we report 5-fold CV MSE. The tables are sorted by the
primary metric (F1 for classification, MSE for regression) so the winner
is visible at the top.

Usage:
    python knn_comparison.py
"""

import time
import numpy as np

from src.methods.knn import KNN
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn


SEED = 100
K_CV = 5
K_VALUES   = [1, 3, 5, 7, 10, 15, 17, 20, 30, 50]
METRICS    = ['l2', 'l1', 'cosine']
WEIGHTINGS = [False, True]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data():
    d = np.load("data/features.npz")
    X_raw = d["xtrain"]
    y_c   = d["ytrainclassif"].astype(int)
    y_r   = d["ytrainreg"]
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y_c, y_r


# --------------------------------------------------------------------------- #
# CV utilities
# --------------------------------------------------------------------------- #

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


def cv_classification(X, y, kwargs, k=K_CV, seed=SEED):
    folds = _stratified_folds(y, k, seed)
    accs, f1s = [], []
    for i in range(k):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = KNN(**kwargs); m.fit(X[tr], y[tr])
        p = m.predict(X[val])
        accs.append(accuracy_fn(p, y[val])); f1s.append(macrof1_fn(p, y[val]))
    return (float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(f1s)), float(np.std(f1s)))


def cv_regression(X, y, kwargs, k=K_CV, seed=SEED):
    folds = _random_folds(len(y), k, seed)
    mses = []
    for i in range(k):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = KNN(**kwargs); m.fit(X[tr], y[tr])
        mses.append(mse_fn(m.predict(X[val]), y[val]))
    return float(np.mean(mses)), float(np.std(mses))


def median_predict_time(kwargs, X, y, repeats=3):
    """KNN fit is O(1) (just stores); predict is O(N²); measure predict on full train."""
    m = KNN(**kwargs); m.fit(X, y)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = m.predict(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0  # ms


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #

def _align(s, w): return s + " " * max(0, w - len(s))


def print_table(rows, headers):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(_align(h, widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(_align(str(r[i]), widths[i]) for i in range(len(headers))))


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def classification_table(X, yc):
    print("\n=== KNN classification — full grid (stratified 5-fold CV) =====================")
    rows = []
    for metric in METRICS:
        for w in WEIGHTINGS:
            for k in K_VALUES:
                acc_m, acc_s, f1_m, f1_s = cv_classification(
                    X, yc, {'k': k, 'task_kind': 'classification',
                            'metric': metric, 'weighted': w})
                rows.append((k, metric, w, acc_m, acc_s, f1_m, f1_s))

    best = max(rows, key=lambda r: r[5])
    print(f"\nBest by macro-F1: k={best[0]}, metric={best[1]}, weighted={best[2]}  "
          f"→ acc={best[3]:.2f}%  F1={best[5]:.3f} ± {best[6]:.3f}")

    # Top-10 by macro-F1
    print("\n-- Top 10 configurations (by macro-F1) --")
    top = sorted(rows, key=lambda r: -r[5])[:10]
    print_table(
        [(str(r[0]), r[1], str(r[2]), f"{r[3]:.2f}%", f"{r[5]:.3f}", f"±{r[6]:.3f}")
         for r in top],
        headers=["k", "metric", "weighted", "acc", "macro-F1", "std"],
    )


def regression_table(X, yr):
    print("\n=== KNN regression — full grid (5-fold CV) ====================================")
    rows = []
    for metric in METRICS:
        for w in WEIGHTINGS:
            for k in K_VALUES:
                mse_m, mse_s = cv_regression(
                    X, yr, {'k': k, 'task_kind': 'regression',
                            'metric': metric, 'weighted': w})
                rows.append((k, metric, w, mse_m, mse_s))

    best = min(rows, key=lambda r: r[3])
    print(f"\nBest by MSE: k={best[0]}, metric={best[1]}, weighted={best[2]}  "
          f"→ MSE={best[3]:.4f} ± {best[4]:.4f}")

    print("\n-- Top 10 configurations (by MSE) --")
    top = sorted(rows, key=lambda r: r[3])[:10]
    print_table(
        [(str(r[0]), r[1], str(r[2]), f"{r[3]:.4f}", f"±{r[4]:.4f}")
         for r in top],
        headers=["k", "metric", "weighted", "MSE", "std"],
    )


def timing_scan(X, y, task_kind='classification'):
    """Predict time (on N training rows) at each metric, k=17, weighted=True."""
    print(f"\n=== KNN predict time — metric comparison ({task_kind}, k=17) =================")
    rows = []
    for metric in METRICS:
        t = median_predict_time(
            {'k': 17, 'task_kind': task_kind, 'metric': metric, 'weighted': True},
            X, y,
        )
        rows.append((metric, f"{t:.1f} ms"))
    print_table(rows, headers=["metric", "predict time"])


# --------------------------------------------------------------------------- #

def main():
    X, yc, yr = load_data()
    print(f"Loaded N = {X.shape[0]}, D = {X.shape[1]} (z-normalized).")
    print(f"Classification class counts: {np.bincount(yc).tolist()}")
    classification_table(X, yc)
    regression_table(X, yr)
    timing_scan(X, yc, 'classification')


if __name__ == "__main__":
    main()
