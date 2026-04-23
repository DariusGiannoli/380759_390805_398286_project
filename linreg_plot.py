"""
Single two-panel figure summarizing every LinearRegression variant:
    (a) median fit time (log scale)
    (b) 5-fold CV MSE (mean ± std)

Both panels share the same y-axis ordering and colour the bars by solver
family (closed-form / gradient descent / polynomial expansion), so the
reader can read off solver and regularization effects in one glance.

Run with:
    python linreg_plot.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.methods.linear_regression import LinearRegression
from src.utils import normalize_fn
from src.evaluation import kfold_cross_validation


VARIANTS = [
    # (display name,                kwargs,                                                  family)
    ("OLS (CF)",                    {'lambda_reg': 0.0,   'method': 'closed_form'},          'cf'),
    ("Ridge λ=10 (CF)",             {'lambda_reg': 10.0,  'method': 'closed_form'},          'cf'),
    ("Ridge λ=100 (CF)",            {'lambda_reg': 100.0, 'method': 'closed_form'},          'cf'),
    ("OLS (GD)",                    {'lambda_reg': 0.0,   'method': 'gradient_descent',
                                     'lr': 0.05, 'max_iters': 2000},                         'gd'),
    ("Ridge λ=10 (GD)",             {'lambda_reg': 10.0,  'method': 'gradient_descent',
                                     'lr': 0.05, 'max_iters': 2000},                         'gd'),
    ("Ridge + poly d=2 (CF)",       {'lambda_reg': 10.0,  'method': 'closed_form',
                                     'degree': 2},                                           'poly'),
    ("Ridge + poly d=2+int (CF)",   {'lambda_reg': 10.0,  'method': 'closed_form',
                                     'degree': 2, 'interaction': True},                     'poly'),
    ("Ridge + poly d=3 (CF)",       {'lambda_reg': 10.0,  'method': 'closed_form',
                                     'degree': 3},                                           'poly'),
]

COLORS = {'cf': '#4C78A8', 'gd': '#F58518', 'poly': '#54A24B'}


def load_data():
    d = np.load("data/features.npz")
    X_raw, y = d["xtrain"], d["ytrainreg"]
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y


def median_fit_time(kwargs, X, y, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        LinearRegression(**kwargs).fit(X, y)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0   # ms


def main():
    X, y = load_data()

    # Fit time
    times_ms = [median_fit_time(kw, X, y) for _, kw, _ in VARIANTS]

    # Train MSE (fit on full data, evaluate on same)
    train_mse = []
    for _, kw, _ in VARIANTS:
        m = LinearRegression(**kw); m.fit(X, y)
        train_mse.append(float(np.mean((m.predict(X) - y) ** 2)))

    # 5-fold CV MSE
    np.random.seed(100)
    _, cv = kfold_cross_validation(LinearRegression,
                                   [kw for _, kw, _ in VARIANTS],
                                   X, y, k=5, task='regression')
    means = [r['mean'] for r in cv]
    stds  = [r['std']  for r in cv]

    names   = [n   for n, _, _ in VARIANTS]
    fams    = [f   for _, _, f in VARIANTS]
    colors  = [COLORS[f] for f in fams]
    y_pos   = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # ---- (a) Fit time ----
    ax = axes[0]
    ax.barh(y_pos, times_ms, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel("fit time (ms, median of 5)")
    ax.set_title("(a) Fit time")
    for i, t in enumerate(times_ms):
        ax.text(t * 1.15, i, f"{t:.2f}", va='center', fontsize=7)
    ax.legend(handles=[
        Patch(facecolor=COLORS['cf'],   label='closed-form'),
        Patch(facecolor=COLORS['gd'],   label='gradient descent'),
        Patch(facecolor=COLORS['poly'], label='polynomial (CF)'),
    ], loc='lower right', fontsize=8, frameon=False)

    # ---- (b) Train MSE vs 5-fold CV MSE ----
    ax = axes[1]
    h = 0.38
    ax.barh(y_pos - h / 2, train_mse, height=h, color=colors,
            edgecolor='black', linewidth=0.4, alpha=0.45,
            label='train MSE')
    ax.barh(y_pos + h / 2, means, height=h, xerr=stds, color=colors,
            edgecolor='black', linewidth=0.4, capsize=3,
            error_kw={'elinewidth': 0.8}, label='5-fold CV MSE')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("MSE  (lighter = train, darker = 5-fold CV)")
    ax.set_title("(b) Train vs CV MSE")
    lo = min(min(train_mse), min(means)) - 0.05
    hi = max(means) + max(stds) + 0.08
    ax.set_xlim(lo, hi)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.005, i + h / 2, f"{m:.3f}",
                va='center', fontsize=7)
    for i, t in enumerate(train_mse):
        ax.text(t + 0.005, i - h / 2, f"{t:.3f}",
                va='center', fontsize=7, alpha=0.7)

    plt.tight_layout()
    plt.savefig("plots/linreg_summary.png", dpi=150, bbox_inches='tight')
    print("Saved plots/linreg_summary.png")


if __name__ == "__main__":
    main()
