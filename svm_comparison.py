"""
Comparison of every LinearSVM variant implemented in this project.

Sweeps over:
  - λ  (L2 strength)           ∈ {1e-4, 1e-3, 1e-2, 1e-1}
  - batch size                 ∈ {full-batch, 64}
  - class weighting            ∈ {None, 'balanced'}
  - lr schedule                ∈ {constant, lr_decay=5e-3}

Reports 5-fold stratified CV accuracy and macro-F1, and the fit time.
Also prints a direct head-to-head with the LogReg baseline at matched
mini-batch settings, so the "same hypothesis class, different loss"
comparison is explicit.

Usage:
    python svm_comparison.py
"""

import time
import numpy as np

from src.methods.extension.svm import LinearSVM
from src.methods.logistic_regression import LogisticRegression
from src.utils import normalize_fn, accuracy_fn, macrof1_fn


SEED = 100
K_CV = 5

BASE = {'lr': 0.05, 'max_iters': 2000, 'beta': 0.9, 'batch_size': 64}

VARIANTS = [
    # name                     kwargs                                                     family
    ("λ=1e-4",                  {**BASE, 'lambda_reg': 1e-4},                              'reg'),
    ("λ=1e-3 *",                {**BASE, 'lambda_reg': 1e-3},                              'reg'),
    ("λ=1e-2",                  {**BASE, 'lambda_reg': 1e-2},                              'reg'),
    ("λ=1e-1",                  {**BASE, 'lambda_reg': 1e-1},                              'reg'),
    ("full-batch (λ=1e-3)",     {'lr': 0.01, 'max_iters': 2000, 'beta': 0.9,
                                 'lambda_reg': 1e-3},                                      'opt'),
    ("+ lr decay γ=5e-3",       {**BASE, 'lambda_reg': 1e-3, 'lr_decay': 5e-3},            'opt'),
    ("+ balanced weights",      {**BASE, 'lambda_reg': 1e-3,
                                 'class_weight': 'balanced'},                              'reg'),
]


# --------------------------------------------------------------------------- #

def load_data():
    d = np.load("data/features.npz")
    X_raw, y = d["xtrain"], d["ytrainclassif"].astype(int)
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y


def _stratified_folds(y, k, seed):
    rng = np.random.RandomState(seed)
    class_idx = [np.where(y == c)[0].copy() for c in np.unique(y)]
    for ci in class_idx:
        rng.shuffle(ci)
    class_folds = [np.array_split(ci, k) for ci in class_idx]
    return [np.concatenate([cf[i] for cf in class_folds]) for i in range(k)]


def cv_classification(cls, X, y, kwargs, k=K_CV, seed=SEED):
    folds = _stratified_folds(y, k, seed)
    accs, f1s = [], []
    for i in range(k):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        np.random.seed(seed + i)
        m = cls(**kwargs); m.fit(X[tr], y[tr])
        p = m.predict(X[val])
        accs.append(accuracy_fn(p, y[val])); f1s.append(macrof1_fn(p, y[val]))
    return (float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(f1s)), float(np.std(f1s)))


def median_fit_time(cls, kwargs, X, y, repeats=3):
    ts = []
    for _ in range(repeats):
        np.random.seed(0)
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

def svm_grid(X, y):
    print("\n=== LinearSVM — variant grid (5-fold stratified CV) ==========================")
    rows = []
    for name, kw, fam in VARIANTS:
        acc_m, acc_s, f1_m, f1_s = cv_classification(LinearSVM, X, y, kw)
        t_ms = median_fit_time(LinearSVM, kw, X, y)
        rows.append((name, fam, acc_m, acc_s, f1_m, f1_s, t_ms))
        print(f"  {name:<25}  acc={acc_m:5.2f}%  F1={f1_m:.3f} "
              f"(±{f1_s:.3f})  time={t_ms:6.1f} ms")

    print("\n-- Sorted by macro-F1 --")
    top = sorted(rows, key=lambda r: -r[4])
    print_table(
        [(r[0], f"{r[2]:.2f}%", f"{r[4]:.3f}", f"±{r[5]:.3f}", f"{r[6]:.0f} ms")
         for r in top],
        headers=["Variant", "acc", "macro-F1", "std", "fit time"],
    )
    best = max(rows, key=lambda r: r[4])
    print(f"\nBest by macro-F1: {best[0]}   acc={best[2]:.2f}%   F1={best[4]:.3f} ± {best[5]:.3f}")


def head_to_head(X, y):
    """Direct SVM vs LogReg comparison at matched mini-batch settings."""
    print("\n=== Head-to-head: LogReg vs LinearSVM (same opt regime, hinge vs log-loss) ===")
    shared_opt = {'lr': 0.1, 'max_iters': 2000, 'beta': 0.9, 'batch_size': 32}
    # LogReg uses lambda_reg=0 (baseline), SVM uses the recommended 1e-3
    lr_kw  = {**shared_opt}
    svm_kw = {'lr': 0.05, 'max_iters': 2000, 'beta': 0.9, 'batch_size': 64,
              'lambda_reg': 1e-3}
    lr_acc, lr_acc_s, lr_f1, lr_f1_s    = cv_classification(LogisticRegression, X, y, lr_kw)
    sv_acc, sv_acc_s, sv_f1, sv_f1_s    = cv_classification(LinearSVM, X, y, svm_kw)
    t_lr  = median_fit_time(LogisticRegression, lr_kw, X, y)
    t_svm = median_fit_time(LinearSVM, svm_kw, X, y)
    rows = [
        ("LogReg (log-loss)", f"{lr_acc:.2f}%",  f"{lr_f1:.3f}", f"±{lr_f1_s:.3f}", f"{t_lr:.0f} ms"),
        ("SVM    (hinge)",    f"{sv_acc:.2f}%",  f"{sv_f1:.3f}", f"±{sv_f1_s:.3f}", f"{t_svm:.0f} ms"),
    ]
    print_table(rows, headers=["Model", "acc", "macro-F1", "std", "fit time"])


def main():
    X, y = load_data()
    print(f"Loaded N = {X.shape[0]}, D = {X.shape[1]} (z-normalized).")
    svm_grid(X, y)
    head_to_head(X, y)


if __name__ == "__main__":
    main()
