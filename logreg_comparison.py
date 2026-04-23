"""
Comparison of every LogisticRegression variant implemented in this project.

Runs stratified 5-fold CV on the classification task for:
  - Vanilla GD baseline + the three enhancement options
    (Xavier init, heavy-ball momentum, early stopping)
  - Inverse-time learning-rate decay
  - Mini-batch SGD (bs=32) and pure SGD (bs=1)
  - L2 regularization at two strengths
  - Inverse-frequency class weighting
  - Polynomial feature expansion at d=2 (per-feature and + interactions) and d=3

Reports per-variant accuracy, macro-F1 (primary metric for this imbalanced
task) and fit time, and prints the table sorted by macro-F1 to identify
the most defensible choice.

Usage:
    python linreg_comparison.py   (LinReg)
    python logreg_comparison.py   (this file)
"""

import time
import numpy as np

from src.methods.logistic_regression import LogisticRegression
from src.utils import normalize_fn, accuracy_fn, macrof1_fn


SEED = 100
K    = 5

# Base config is the "selected" optimizer: full-batch GD, η=0.3, 1000 iters,
# β=0.9 momentum, zeros init, no L2, no class weighting, no poly.
BASE = {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9}

VARIANTS = [
    # name                       kwargs (added on top of / replacing BASE)     family
    ("vanilla GD",                {'lr': 0.3, 'max_iters': 1000},              'opt'),
    ("+ Xavier init",             {**BASE, 'init': 'xavier'},                  'opt'),
    ("+ momentum β=0.9 *",        {**BASE},                                    'opt'),
    ("+ early stop (1e-6)",       {**BASE, 'tol': 1e-6},                       'opt'),
    ("+ lr decay γ=5e-3",         {'lr': 0.5, 'max_iters': 1000, 'beta': 0.9,
                                   'lr_decay': 5e-3},                          'opt'),
    ("mini-batch bs=32",          {'lr': 0.1, 'max_iters': 2000, 'beta': 0.9,
                                   'batch_size': 32},                          'opt'),
    ("SGD bs=1",                  {'lr': 0.05, 'max_iters': 2000, 'beta': 0.9,
                                   'batch_size': 1},                           'opt'),
    ("+ L2 λ=10",                 {**BASE, 'lambda_reg': 10.0},                'reg'),
    ("+ L2 λ=100",                {**BASE, 'lambda_reg': 100.0},               'reg'),
    ("+ balanced class weights",  {**BASE, 'class_weight': 'balanced'},        'reg'),
    ("+ poly d=2",                {**BASE, 'degree': 2},                       'poly'),
    ("+ poly d=2+int",            {**BASE, 'degree': 2, 'interaction': True},  'poly'),
    ("+ poly d=3",                {**BASE, 'degree': 3},                       'poly'),
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_classification_data():
    d = np.load("data/features.npz")
    X_raw, y = d["xtrain"], d["ytrainclassif"].astype(int)
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y


# --------------------------------------------------------------------------- #
# Stratified k-fold CV returning both accuracy and macro-F1
# --------------------------------------------------------------------------- #

def stratified_cv_acc_f1(X, y, kwargs, k=K, seed=SEED):
    rng = np.random.RandomState(seed)
    class_idx = [np.where(y == c)[0].copy() for c in np.unique(y)]
    for ci in class_idx:
        rng.shuffle(ci)
    class_folds = [np.array_split(ci, k) for ci in class_idx]
    folds = [np.concatenate([cf[i] for cf in class_folds]) for i in range(k)]

    accs, f1s = [], []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        np.random.seed(seed + i)   # reproducible weight inits / SGD shuffles
        model = LogisticRegression(**kwargs)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        accs.append(accuracy_fn(preds, y[val_idx]))
        f1s.append(macrof1_fn(preds, y[val_idx]))
    return (float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(f1s)), float(np.std(f1s)))


def median_fit_time(kwargs, X, y, repeats=3):
    ts = []
    for _ in range(repeats):
        np.random.seed(0)
        t0 = time.perf_counter()
        LogisticRegression(**kwargs).fit(X, y)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0   # ms


# --------------------------------------------------------------------------- #
# Table printing
# --------------------------------------------------------------------------- #

def _align(s, w):
    return s + " " * max(0, w - len(s))


def print_table(rows, headers):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(_align(h, widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(_align(str(r[i]), widths[i]) for i in range(len(headers))))


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def main():
    X, y = load_classification_data()
    print(f"Loaded N = {X.shape[0]}, D = {X.shape[1]} (z-normalized).")
    print(f"Class counts: {np.bincount(y).tolist()}  (stratified 5-fold CV)\n")

    rows = []
    for name, kw, fam in VARIANTS:
        acc_mu, acc_sd, f1_mu, f1_sd = stratified_cv_acc_f1(X, y, kw)
        t_ms = median_fit_time(kw, X, y)
        rows.append((name, fam, acc_mu, acc_sd, f1_mu, f1_sd, t_ms))
        print(f"  {name:<28}  acc={acc_mu:5.2f}%  F1={f1_mu:.3f}  "
              f"(±{f1_sd:.3f})  time={t_ms:6.1f} ms")

    # Pretty table sorted by macro-F1 (higher is better for F1)
    print("\n-- Sorted by macro-F1 (high → low) --")
    display = sorted(
        rows, key=lambda r: -r[4]
    )
    print_table(
        [(r[0], f"{r[2]:.2f}%", f"{r[4]:.3f}", f"±{r[5]:.3f}", f"{r[6]:.1f} ms")
         for r in display],
        headers=["Variant", "acc", "macro-F1", "std", "fit time"],
    )

    best = max(rows, key=lambda r: r[4])
    print(f"\nBest F1 variant: {best[0]}   "
          f"acc = {best[2]:.2f}%   macro-F1 = {best[4]:.3f} ± {best[5]:.3f}")


if __name__ == "__main__":
    main()
