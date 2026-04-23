"""
Single two-panel figure summarizing every LogisticRegression variant:
    (a) median fit time (log scale)
    (b) classification accuracy and macro-F1 (5-fold stratified CV)

Both panels share the same y-axis ordering and colour the bars by variant
family (optimizer / regularization / polynomial), so the reader can read
off solver cost and the accuracy-vs-F1 tradeoff in one glance.

Run with:
    python logreg_plot.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.methods.logistic_regression import LogisticRegression
from src.utils import normalize_fn, accuracy_fn, macrof1_fn


BASE = {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9}

VARIANTS = [
    # name                          kwargs                                                    family
    ("vanilla GD",                  {'lr': 0.3, 'max_iters': 1000},                           'opt'),
    ("+ Xavier init",               {**BASE, 'init': 'xavier'},                               'opt'),
    ("+ momentum β=0.9 *",          {**BASE},                                                 'opt'),
    ("+ early stop (1e-6)",         {**BASE, 'tol': 1e-6},                                    'opt'),
    ("+ lr decay γ=5e-3",           {'lr': 0.5, 'max_iters': 1000, 'beta': 0.9,
                                     'lr_decay': 5e-3},                                        'opt'),
    ("mini-batch bs=32",            {'lr': 0.1, 'max_iters': 2000, 'beta': 0.9,
                                     'batch_size': 32},                                        'opt'),
    ("SGD bs=1",                    {'lr': 0.05, 'max_iters': 2000, 'beta': 0.9,
                                     'batch_size': 1},                                         'opt'),
    ("+ L2 λ=10",                   {**BASE, 'lambda_reg': 10.0},                             'reg'),
    ("+ L2 λ=100",                  {**BASE, 'lambda_reg': 100.0},                            'reg'),
    ("+ balanced class weights",    {**BASE, 'class_weight': 'balanced'},                     'reg'),
    ("+ poly d=2",                  {**BASE, 'degree': 2},                                    'poly'),
    ("+ poly d=2+int",              {**BASE, 'degree': 2, 'interaction': True},               'poly'),
    ("+ poly d=3",                  {**BASE, 'degree': 3},                                    'poly'),
]

COLORS = {'opt': '#4C78A8', 'reg': '#F58518', 'poly': '#54A24B'}


def load_data():
    d = np.load("data/features.npz")
    X_raw, y = d["xtrain"], d["ytrainclassif"].astype(int)
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y


def stratified_cv_acc_f1(X, y, kwargs, k=5, seed=100):
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
        np.random.seed(seed + i)
        m = LogisticRegression(**kwargs)
        m.fit(X[train_idx], y[train_idx])
        p = m.predict(X[val_idx])
        accs.append(accuracy_fn(p, y[val_idx]))
        f1s.append(macrof1_fn(p, y[val_idx]))
    return (float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(f1s)), float(np.std(f1s)))


def median_fit_time(kwargs, X, y, repeats=3):
    ts = []
    for _ in range(repeats):
        np.random.seed(0)
        t0 = time.perf_counter()
        LogisticRegression(**kwargs).fit(X, y)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0


def main():
    X, y = load_data()

    times_ms, acc_m, acc_s, f1_m, f1_s = [], [], [], [], []
    for _, kw, _ in VARIANTS:
        am, asd, fm, fsd = stratified_cv_acc_f1(X, y, kw)
        acc_m.append(am); acc_s.append(asd); f1_m.append(fm); f1_s.append(fsd)
        times_ms.append(median_fit_time(kw, X, y))

    names  = [n for n, _, _ in VARIANTS]
    fams   = [f for _, _, f in VARIANTS]
    colors = [COLORS[f] for f in fams]
    y_pos  = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

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
        Patch(facecolor=COLORS['opt'],  label='optimizer'),
        Patch(facecolor=COLORS['reg'],  label='regularization'),
        Patch(facecolor=COLORS['poly'], label='polynomial'),
    ], loc='lower right', fontsize=8, frameon=False)

    # ---- (b) Accuracy vs macro-F1 ----
    ax = axes[1]
    h = 0.38
    # Accuracy in %, convert to 0-1 for shared axis with F1
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
    plt.savefig("plots/logreg_summary.png", dpi=150, bbox_inches='tight')
    print("Saved plots/logreg_summary.png")


if __name__ == "__main__":
    main()
