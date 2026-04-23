"""
Single two-panel figure summarizing every kernel configuration:
    (a) KRR  (regression) 5-fold CV MSE  — lower is better
    (b) KLS  (classification) accuracy + macro-F1 — higher is better

Bars grouped by kernel family (linear / polynomial / RBF) so the reader
sees the structural story at a glance:
  - linear kernel ties LinReg on regression
  - RBF needs a wide σ to match it (large σ → near-linear)
  - polynomial d=3 overfits both tasks
  - KLS-linear underperforms because one-hot KLS has no per-class bias

Run with:
    python kernel_ridge_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.methods.extension.kernel_ridge import (
    KernelRidgeRegression, KernelLeastSquaresClassifier,
)
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn


SEED = 100

KERNELS = [
    # (display name, kwargs, family)
    ("linear",         {'kernel': 'linear'},                           'lin'),
    ("poly d=2",       {'kernel': 'poly', 'degree': 2, 'coef0': 1.0},  'poly'),
    ("poly d=3",       {'kernel': 'poly', 'degree': 3, 'coef0': 1.0},  'poly'),
    ("rbf σ=1",        {'kernel': 'rbf', 'sigma': 1.0},                'rbf'),
    ("rbf σ=3",        {'kernel': 'rbf', 'sigma': 3.0},                'rbf'),
    ("rbf σ=5",        {'kernel': 'rbf', 'sigma': 5.0},                'rbf'),
    ("rbf σ=10",       {'kernel': 'rbf', 'sigma': 10.0},               'rbf'),
]

COLORS = {'lin': '#4C78A8', 'poly': '#54A24B', 'rbf': '#F58518'}


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


def cv_krr(X, y, kwargs, k=5):
    folds = _random_folds(len(y), k, SEED)
    mses = []
    for i in range(k):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = KernelRidgeRegression(**kwargs); m.fit(X[tr], y[tr])
        mses.append(mse_fn(m.predict(X[val]), y[val]))
    return float(np.mean(mses)), float(np.std(mses))


def cv_kls(X, y, kwargs, k=5):
    folds = _stratified_folds(y, k, SEED)
    accs, f1s = [], []
    for i in range(k):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = KernelLeastSquaresClassifier(**kwargs); m.fit(X[tr], y[tr])
        p = m.predict(X[val])
        accs.append(accuracy_fn(p, y[val])); f1s.append(macrof1_fn(p, y[val]))
    return (float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(f1s)), float(np.std(f1s)))


def main():
    X, yc, yr = load_data()

    krr_mse_m, krr_mse_s, kls_acc_m, kls_f1_m, kls_f1_s = [], [], [], [], []
    for _, kw, _ in KERNELS:
        mse_m, mse_s = cv_krr(X, yr, {'lambda_reg': 1.0, **kw})
        krr_mse_m.append(mse_m); krr_mse_s.append(mse_s)
        acc_m, _, f1_m, f1_s = cv_kls(X, yc, {'lambda_reg': 1.0, **kw})
        kls_acc_m.append(acc_m); kls_f1_m.append(f1_m); kls_f1_s.append(f1_s)

    names  = [n for n, _, _ in KERNELS]
    fams   = [f for _, _, f in KERNELS]
    colors = [COLORS[f] for f in fams]
    y_pos  = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # ---- (a) KRR regression MSE ----
    ax = axes[0]
    ax.barh(y_pos, krr_mse_m, xerr=krr_mse_s, color=colors,
            edgecolor='black', linewidth=0.4, capsize=3,
            error_kw={'elinewidth': 0.8})
    ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV MSE (mean ± std)")
    ax.set_title("(a) KRR — regression")
    for i, (m, s) in enumerate(zip(krr_mse_m, krr_mse_s)):
        ax.text(m + s + 0.03, i, f"{m:.3f}", va='center', fontsize=7)
    ax.legend(handles=[
        Patch(facecolor=COLORS['lin'],  label='linear'),
        Patch(facecolor=COLORS['poly'], label='polynomial'),
        Patch(facecolor=COLORS['rbf'],  label='RBF'),
    ], loc='lower right', fontsize=8, frameon=False)

    # ---- (b) KLS classification accuracy + macro-F1 ----
    ax = axes[1]
    h = 0.38
    acc_frac = [a / 100.0 for a in kls_acc_m]
    ax.barh(y_pos - h / 2, acc_frac, height=h, color=colors,
            edgecolor='black', linewidth=0.4, alpha=0.45,
            label='accuracy')
    ax.barh(y_pos + h / 2, kls_f1_m, height=h, xerr=kls_f1_s, color=colors,
            edgecolor='black', linewidth=0.4, capsize=3,
            error_kw={'elinewidth': 0.8}, label='macro-F1')
    ax.set_yticks(y_pos); ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("score  (lighter = accuracy, darker = macro-F1)")
    ax.set_title("(b) KLS — classification")
    ax.set_xlim(min(min(kls_f1_m), min(acc_frac)) - 0.05,
                max(max(kls_f1_m) + max(kls_f1_s), max(acc_frac)) + 0.05)
    for i, (a, f) in enumerate(zip(acc_frac, kls_f1_m)):
        ax.text(a + 0.005, i - h / 2, f"{a * 100:.1f}%",
                va='center', fontsize=7, alpha=0.7)
        ax.text(f + kls_f1_s[i] + 0.005, i + h / 2, f"{f:.3f}",
                va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig("plots/kernel_ridge_summary.png", dpi=150, bbox_inches='tight')
    print("Saved plots/kernel_ridge_summary.png")


if __name__ == "__main__":
    main()
