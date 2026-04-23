"""
Single two-panel figure summarizing every LinearSVM variant, with a
LogReg baseline bar for the "hinge vs log-loss" head-to-head:

    (a) median fit time (log scale)
    (b) classification accuracy and macro-F1 (5-fold stratified CV)

Run with:
    python svm_plot.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.methods.extension.svm import LinearSVM
from src.methods.logistic_regression import LogisticRegression
from src.utils import normalize_fn, accuracy_fn, macrof1_fn


SEED = 100
BASE = {'lr': 0.05, 'max_iters': 2000, 'beta': 0.9, 'batch_size': 64}

# Each entry: (display name, model class, kwargs, family)
VARIANTS = [
    ("SVM λ=1e-4",                 LinearSVM,          {**BASE, 'lambda_reg': 1e-4}, 'reg'),
    ("SVM λ=1e-3 *",               LinearSVM,          {**BASE, 'lambda_reg': 1e-3}, 'reg'),
    ("SVM λ=1e-2",                 LinearSVM,          {**BASE, 'lambda_reg': 1e-2}, 'reg'),
    ("SVM λ=1e-1",                 LinearSVM,          {**BASE, 'lambda_reg': 1e-1}, 'reg'),
    ("SVM full-batch",             LinearSVM,          {'lr': 0.01, 'max_iters': 2000,
                                                         'beta': 0.9, 'lambda_reg': 1e-3}, 'opt'),
    ("SVM + lr decay",             LinearSVM,          {**BASE, 'lambda_reg': 1e-3,
                                                         'lr_decay': 5e-3},            'opt'),
    ("SVM + balanced weights",     LinearSVM,          {**BASE, 'lambda_reg': 1e-3,
                                                         'class_weight': 'balanced'},  'reg'),
    ("LogReg (baseline)",          LogisticRegression, {'lr': 0.1, 'max_iters': 2000,
                                                         'beta': 0.9, 'batch_size': 32}, 'logreg'),
]

COLORS = {'reg': '#F58518', 'opt': '#4C78A8', 'logreg': '#9E6FB0'}


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


def cv_acc_f1(cls, X, y, kwargs, k=5, seed=SEED):
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


def main():
    X, y = load_data()

    times_ms, acc_m, acc_s, f1_m, f1_s = [], [], [], [], []
    for _, cls, kw, _ in VARIANTS:
        am, asd, fm, fsd = cv_acc_f1(cls, X, y, kw)
        acc_m.append(am); acc_s.append(asd); f1_m.append(fm); f1_s.append(fsd)
        times_ms.append(median_fit_time(cls, kw, X, y))

    names  = [n for n, _, _, _ in VARIANTS]
    fams   = [f for _, _, _, f in VARIANTS]
    colors = [COLORS[f] for f in fams]
    y_pos  = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # ---- (a) Fit time ----
    ax = axes[0]
    ax.barh(y_pos, times_ms, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel("fit time (ms, median of 3)")
    ax.set_title("(a) Fit time")
    for i, t in enumerate(times_ms):
        ax.text(t * 1.15, i, f"{t:.0f}", va='center', fontsize=7)
    ax.legend(handles=[
        Patch(facecolor=COLORS['reg'],    label='SVM regularization'),
        Patch(facecolor=COLORS['opt'],    label='SVM optimizer'),
        Patch(facecolor=COLORS['logreg'], label='LogReg baseline'),
    ], loc='lower right', fontsize=8, frameon=False)

    # ---- (b) Accuracy vs macro-F1 ----
    ax = axes[1]
    h = 0.38
    acc_frac = [a / 100.0 for a in acc_m]
    ax.barh(y_pos - h / 2, acc_frac, height=h, color=colors,
            edgecolor='black', linewidth=0.4, alpha=0.45,
            label='accuracy')
    ax.barh(y_pos + h / 2, f1_m, height=h, xerr=f1_s, color=colors,
            edgecolor='black', linewidth=0.4, capsize=3,
            error_kw={'elinewidth': 0.8}, label='macro-F1')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("score  (lighter = accuracy, darker = macro-F1)")
    ax.set_title("(b) Accuracy vs macro-F1")
    ax.set_xlim(min(min(f1_m), min(acc_frac)) - 0.05,
                max(max(f1_m) + max(f1_s), max(acc_frac)) + 0.05)
    for i, (a, f) in enumerate(zip(acc_frac, f1_m)):
        ax.text(a + 0.005, i - h / 2, f"{a * 100:.1f}%",
                va='center', fontsize=7, alpha=0.7)
        ax.text(f + f1_s[i] + 0.005, i + h / 2, f"{f:.3f}",
                va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig("plots/svm_summary.png", dpi=150, bbox_inches='tight')
    print("Saved plots/svm_summary.png")


if __name__ == "__main__":
    main()
