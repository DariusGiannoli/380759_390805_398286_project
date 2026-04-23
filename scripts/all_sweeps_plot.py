"""
Generate the 2x2 hyperparameter-sweep figure used in the report
(§2.2 Cross-Validation, Fig. 1):
    (a) Linear Regression ridge λ sweep    (log-x, MSE)
    (b) Logistic Regression lr sweep       (log-x, 1-acc)
    (c) KNN k sweep, classification        (1-acc)
    (d) KNN k sweep, regression            (MSE)

Saves plots/all_sweeps.png at dpi=150.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.linear_regression import LinearRegression
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.utils import normalize_fn
from src.optimization import hyperparameter_sweep


SEED = 100


def load_data():
    d = np.load("data/features.npz")
    X_raw = d["xtrain"]
    y_c   = d["ytrainclassif"].astype(int)
    y_r   = d["ytrainreg"]
    mean, std = X_raw.mean(0), X_raw.std(0)
    std[std == 0] = 1
    return normalize_fn(X_raw, mean, std), y_c, y_r


def _plot_sweep_panel(ax, sweep, ylab, title, log_x=False):
    x = sweep['param_values']
    ax.errorbar(x, sweep['means'], yerr=sweep['stds'],
                marker='o', capsize=4)
    if log_x:
        ax.set_xscale('log')
    best = int(np.argmin(sweep['means']))
    ax.scatter([x[best]], [sweep['means'][best]],
               color='red', s=80, zorder=3,
               label=f"best: {sweep['param_name']}={x[best]}")
    ax.set_xlabel(sweep['param_name'])
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()


def main():
    X, yc, yr = load_data()
    os.makedirs("plots", exist_ok=True)

    np.random.seed(SEED)
    linreg_lambda_sweep = hyperparameter_sweep(
        LinearRegression, 'lambda_reg',
        [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        fixed_kwargs={}, features=X, labels=yr,
        task='regression', k=5,
    )

    logreg_lr_sweep = hyperparameter_sweep(
        LogisticRegression, 'lr',
        [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
        fixed_kwargs={'max_iters': 1000},
        features=X, labels=yc, task='classification', k=5,
    )

    knn_sweep_cl = hyperparameter_sweep(
        KNN, 'k', [1, 3, 5, 7, 10, 15, 17, 20, 30, 50],
        fixed_kwargs={'task_kind': 'classification',
                      'metric': 'cosine', 'weighted': True},
        features=X, labels=yc, task='classification', k=5,
    )

    knn_sweep_rg = hyperparameter_sweep(
        KNN, 'k', [1, 3, 5, 7, 10, 14, 15, 20, 30, 50],
        fixed_kwargs={'task_kind': 'regression',
                      'metric': 'cosine', 'weighted': True},
        features=X, labels=yr, task='regression', k=5,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    _plot_sweep_panel(axes[0, 0], linreg_lambda_sweep, "MSE",
                      "(a) Linear Regression — λ", log_x=True)
    _plot_sweep_panel(axes[0, 1], logreg_lr_sweep, "Error rate (1 - acc)",
                      "(b) Logistic Regression — learning rate", log_x=True)
    _plot_sweep_panel(axes[1, 0], knn_sweep_cl, "Error rate (1 - acc)",
                      "(c) KNN classification — k  (cosine, weighted)")
    _plot_sweep_panel(axes[1, 1], knn_sweep_rg, "MSE",
                      "(d) KNN regression — k  (cosine, weighted)")
    plt.tight_layout()
    plt.savefig("plots/all_sweeps.png", dpi=150, bbox_inches="tight")
    print("Saved plots/all_sweeps.png")


if __name__ == "__main__":
    main()
