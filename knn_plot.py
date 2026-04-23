"""
Single two-panel figure summarizing KNN across both tasks:
    (a) classification CV macro-F1 vs k  (3 metrics, inverse-distance weighted)
    (b) regression     CV MSE     vs k  (3 metrics, inverse-distance weighted)

Inverse-distance weighting beats uniform by ~0.02 F1 / 0.02 MSE across
the board on this dataset (verified separately); we show only weighted
curves here to keep the figure legible.

Run with:
    python knn_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt

from src.methods.knn import KNN
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn


SEED = 100
K_VALUES = [1, 3, 5, 7, 10, 15, 17, 20, 30, 50]
METRICS  = ['l2', 'l1', 'cosine']
COLORS   = {'l2': '#4C78A8', 'l1': '#F58518', 'cosine': '#54A24B'}


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


def cv_classification(X, y, kwargs):
    folds = _stratified_folds(y, 5, SEED)
    f1s = []
    for i in range(5):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        m = KNN(**kwargs); m.fit(X[tr], y[tr])
        f1s.append(macrof1_fn(m.predict(X[val]), y[val]))
    return float(np.mean(f1s)), float(np.std(f1s))


def cv_regression(X, y, kwargs):
    folds = _random_folds(len(y), 5, SEED)
    mses = []
    for i in range(5):
        val = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        m = KNN(**kwargs); m.fit(X[tr], y[tr])
        mses.append(mse_fn(m.predict(X[val]), y[val]))
    return float(np.mean(mses)), float(np.std(mses))


def main():
    X, yc, yr = load_data()

    # --- Compute CV F1 and MSE grids ---
    f1_grid  = {m: {'mean': [], 'std': []} for m in METRICS}
    mse_grid = {m: {'mean': [], 'std': []} for m in METRICS}
    for metric in METRICS:
        for k in K_VALUES:
            f1_m, f1_s = cv_classification(
                X, yc, {'k': k, 'task_kind': 'classification',
                        'metric': metric, 'weighted': True})
            f1_grid[metric]['mean'].append(f1_m)
            f1_grid[metric]['std'].append(f1_s)
            mse_m, mse_s = cv_regression(
                X, yr, {'k': k, 'task_kind': 'regression',
                        'metric': metric, 'weighted': True})
            mse_grid[metric]['mean'].append(mse_m)
            mse_grid[metric]['std'].append(mse_s)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ks = np.array(K_VALUES)

    # ---- (a) Classification ----
    ax = axes[0]
    for metric in METRICS:
        mu = np.array(f1_grid[metric]['mean']); sd = np.array(f1_grid[metric]['std'])
        ax.errorbar(ks, mu, yerr=sd, marker='o', markersize=4,
                    color=COLORS[metric], label=metric, capsize=2, linewidth=1.2)
    # Best marker
    best_metric = max(METRICS, key=lambda m: max(f1_grid[m]['mean']))
    best_mu = np.array(f1_grid[best_metric]['mean'])
    best_idx = int(np.argmax(best_mu))
    ax.scatter([ks[best_idx]], [best_mu[best_idx]], color='red', s=80,
               zorder=3, label=f"best: k={ks[best_idx]}, {best_metric}")
    ax.set_xlabel("k")
    ax.set_ylabel("macro-F1 (5-fold CV)")
    ax.set_title("(a) Classification — F1 vs k  (weighted)")
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)

    # ---- (b) Regression ----
    ax = axes[1]
    for metric in METRICS:
        mu = np.array(mse_grid[metric]['mean']); sd = np.array(mse_grid[metric]['std'])
        ax.errorbar(ks, mu, yerr=sd, marker='o', markersize=4,
                    color=COLORS[metric], label=metric, capsize=2, linewidth=1.2)
    best_metric = min(METRICS, key=lambda m: min(mse_grid[m]['mean']))
    best_mu = np.array(mse_grid[best_metric]['mean'])
    best_idx = int(np.argmin(best_mu))
    ax.scatter([ks[best_idx]], [best_mu[best_idx]], color='red', s=80,
               zorder=3, label=f"best: k={ks[best_idx]}, {best_metric}")
    ax.set_xlabel("k")
    ax.set_ylabel("MSE (5-fold CV)")
    ax.set_title("(b) Regression — MSE vs k  (weighted)")
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/knn_summary.png", dpi=150, bbox_inches='tight')
    print("Saved plots/knn_summary.png")


if __name__ == "__main__":
    main()
