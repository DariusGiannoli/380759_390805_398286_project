"""
Regenerates all plots needed for the report using the final tuned
hyperparameters.

Run with:
    python generate_report_plots.py
"""

import os
import numpy as np

from src.methods.linear_regression import LinearRegression
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.utils import normalize_fn
from src.evaluation import (
    plot_confusion_matrix,
    compute_learning_curve, plot_learning_curve,
    timing_comparison, plot_timing,
    plot_loss_curve,
)
from src.optimization import (
    hyperparameter_sweep, plot_hyperparameter_sweep,
    ablation_comparison, plot_ablation,
    compare_cv_methods, plot_cv_comparison,
)

np.random.seed(100)
os.makedirs("plots", exist_ok=True)


# -----------------------------------------------------------------------------
# FINAL TUNED HYPERPARAMETERS (from coarse-to-fine search)
# -----------------------------------------------------------------------------
BEST_LINREG       = {'lambda_reg': 10, 'degree': 1, 'interaction': False}
BEST_LOGREG       = {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9, 'init': 'zeros'}
BEST_KNN_CLASSIF  = {'k': 17, 'task_kind': 'classification',
                     'metric': 'cosine', 'weighted': True}
BEST_KNN_REG      = {'k': 14, 'task_kind': 'regression',
                     'metric': 'cosine', 'weighted': True}


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
data = np.load("data/features.npz", allow_pickle=True)
Xtr, Xte = data['xtrain'], data['xtest']
ytr_cl, yte_cl = data['ytrainclassif'], data['ytestclassif']
ytr_rg, yte_rg = data['ytrainreg'],     data['ytestreg']

mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1
Xtr = normalize_fn(Xtr, mean, std)
Xte = normalize_fn(Xte, mean, std)


# =============================================================================
# LINEAR REGRESSION
# =============================================================================
print("\n" + "=" * 60)
print("  LINEAR REGRESSION")
print("=" * 60)

# lambda sweep — log-scaled (baseline vanilla linear, no polynomial)
linreg_lambda_sweep = hyperparameter_sweep(
    LinearRegression, 'lambda_reg',
    [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    fixed_kwargs={}, features=Xtr, labels=ytr_rg,
    task='regression', k=5,
)
plot_hyperparameter_sweep(linreg_lambda_sweep, 'regression', log_x=True,
                          save_path="plots/linreg_lambda_sweep.png")

# ablation: polynomial + regularization
ablation = ablation_comparison(
    LinearRegression,
    configs=[
        ("vanilla",         {}),
        ("L2 reg λ=10 *",   {'lambda_reg': 10}),
        ("degree=2",        {'degree': 2}),
        ("degree=2 + int.", {'degree': 2, 'interaction': True}),
        ("degree=3",        {'degree': 3}),
    ],
    features=Xtr, labels=ytr_rg, task='regression', k=5,
)
plot_ablation(ablation, 'regression',
              title="Linear Regression ablation",
              save_path="plots/linreg_ablation.png")

# CV method comparison — val vs k-fold vs LOOCV
cv_cmp = compare_cv_methods(
    Xtr, ytr_rg,
    lambdas=[0, 0.001, 0.01, 0.1, 1, 10, 100], k=5,
)
plot_cv_comparison(cv_cmp, save_path="plots/linreg_cv_comparison.png")

# learning curve with BEST HPs
lc = compute_learning_curve(
    LinearRegression, BEST_LINREG, Xtr, ytr_rg, 'regression',
)
plot_learning_curve(lc, 'regression',
                    title="Linear Regression — learning curve",
                    save_path="plots/linreg_learning_curve.png")


# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
print("\n" + "=" * 60)
print("  LOGISTIC REGRESSION")
print("=" * 60)

# lr sweep (log-scaled)
logreg_lr_sweep = hyperparameter_sweep(
    LogisticRegression, 'lr',
    [1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0],
    fixed_kwargs={'max_iters': 1000, 'beta': 0.9, 'init': 'zeros'},
    features=Xtr, labels=ytr_cl, task='classification', k=5,
)
plot_hyperparameter_sweep(logreg_lr_sweep, 'classification', log_x=True,
                          save_path="plots/logreg_lr_sweep.png")

# ablation — each row shares the same lr/iters so it isolates the enhancement
ablation = ablation_comparison(
    LogisticRegression,
    configs=[
        ("vanilla GD (zeros)", {'lr': 0.3, 'max_iters': 1000}),
        ("xavier init",        {'lr': 0.3, 'max_iters': 1000, 'init': 'xavier'}),
        ("+ momentum β=0.9",   {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9}),
        ("+ early stop",       {'lr': 0.3, 'max_iters': 1000, 'tol': 1e-6}),
        ("all enhancements*",  {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9,
                                'tol': 1e-6, 'init': 'xavier'}),
    ],
    features=Xtr, labels=ytr_cl, task='classification', k=5,
)
plot_ablation(ablation, 'classification',
              title="Logistic Regression ablation",
              save_path="plots/logreg_ablation.png")

# learning curve with BEST HPs
lc = compute_learning_curve(
    LogisticRegression, BEST_LOGREG, Xtr, ytr_cl, 'classification',
)
plot_learning_curve(lc, 'classification',
                    title="Logistic Regression — learning curve",
                    save_path="plots/logreg_learning_curve.png")

# train a single LogReg with BEST HPs — reused for loss curve + confusion matrix
m = LogisticRegression(**BEST_LOGREG)
m.fit(Xtr, ytr_cl)

# training loss curve (convergence diagnostic)
plot_loss_curve(m.loss_history, save_path="plots/logreg_loss_curve.png")

# confusion matrix on TEST SET
plot_confusion_matrix(m.predict(Xte), yte_cl,
                      class_names=['Low', 'Medium', 'High'],
                      save_path="plots/logreg_confusion_matrix.png")


# =============================================================================
# KNN
# =============================================================================
print("\n" + "=" * 60)
print("  KNN")
print("=" * 60)

# k sweep — classification (with BEST metric + weighting)
sweep = hyperparameter_sweep(
    KNN, 'k', [1, 3, 5, 7, 10, 15, 17, 20, 30, 50],
    fixed_kwargs={'task_kind': 'classification',
                  'metric': 'cosine', 'weighted': True},
    features=Xtr, labels=ytr_cl, task='classification', k=5,
)
plot_hyperparameter_sweep(sweep, 'classification', log_x=False,
                          save_path="plots/knn_k_classif.png")

# k sweep — regression
sweep = hyperparameter_sweep(
    KNN, 'k', [1, 3, 5, 7, 10, 14, 15, 20, 30, 50],
    fixed_kwargs={'task_kind': 'regression',
                  'metric': 'cosine', 'weighted': True},
    features=Xtr, labels=ytr_rg, task='regression', k=5,
)
plot_hyperparameter_sweep(sweep, 'regression', log_x=False,
                          save_path="plots/knn_k_reg.png")

# ablation — metric × weighted (fix k=17, the classification winner)
ablation = ablation_comparison(
    KNN,
    configs=[
        ("l2, uniform",    {'k': 17, 'task_kind': 'classification'}),
        ("l1, uniform",    {'k': 17, 'task_kind': 'classification',
                            'metric': 'l1'}),
        ("cosine, uniform",{'k': 17, 'task_kind': 'classification',
                            'metric': 'cosine'}),
        ("l2, weighted",   {'k': 17, 'task_kind': 'classification',
                            'weighted': True}),
        ("cosine+weighted*",{'k': 17, 'task_kind': 'classification',
                             'metric': 'cosine', 'weighted': True}),
    ],
    features=Xtr, labels=ytr_cl, task='classification', k=5,
)
plot_ablation(ablation, 'classification',
              title="KNN ablation (classification)",
              save_path="plots/knn_ablation.png")

# confusion matrix for KNN classif on TEST SET with BEST HPs
m = KNN(**BEST_KNN_CLASSIF)
m.fit(Xtr, ytr_cl)
plot_confusion_matrix(m.predict(Xte), yte_cl,
                      class_names=['Low', 'Medium', 'High'],
                      save_path="plots/knn_confusion_matrix.png")

# timing scaling with BEST HPs
timing = timing_comparison(KNN, BEST_KNN_CLASSIF, Xtr, ytr_cl)
plot_timing(timing, title="KNN timing vs training size",
            save_path="plots/knn_timing.png")


# --- Combined timing for all three methods at tuned HPs --------------------
import time as _time
import matplotlib.pyplot as _plt

_SPECS = [
    ("Linear Regression",   LinearRegression,   BEST_LINREG,      ytr_rg, "o-"),
    ("Logistic Regression", LogisticRegression, BEST_LOGREG,      ytr_cl, "s-"),
    ("KNN (k=17)",          KNN,                BEST_KNN_CLASSIF, ytr_cl, "^-"),
]
_N        = len(Xtr)
_sizes    = np.linspace(100, _N, 8, dtype=int)
_repeats  = 3


def _measure(cls, kwargs, X, y):
    fit_t, pred_t = [], []
    for rep in range(_repeats):
        rng  = np.random.RandomState(rep)
        perm = rng.permutation(len(X))
        row_f, row_p = [], []
        for s in _sizes:
            Xs, ys = X[perm[:s]], y[perm[:s]]
            m = cls(**kwargs)
            t0 = _time.time(); m.fit(Xs, ys);  row_f.append(_time.time() - t0)
            t0 = _time.time(); m.predict(Xs); row_p.append(_time.time() - t0)
        fit_t.append(row_f); pred_t.append(row_p)
    fit_t, pred_t = np.array(fit_t) * 1e3, np.array(pred_t) * 1e3
    return fit_t.mean(0), fit_t.std(0), pred_t.mean(0), pred_t.std(0)


_fig, _axes = _plt.subplots(1, 2, figsize=(10, 3.8))
for _name, _cls, _kw, _y, _mk in _SPECS:
    print(f"Timing {_name} ...")
    _fm, _fs, _pm, _ps = _measure(_cls, _kw, Xtr, _y)
    _axes[0].errorbar(_sizes, _fm, yerr=_fs, fmt=_mk, capsize=3, label=_name)
    _axes[1].errorbar(_sizes, _pm, yerr=_ps, fmt=_mk, capsize=3, label=_name)

for _ax, _ttl in zip(_axes, ["(a) Fit time", "(b) Predict time"]):
    _ax.set_xlabel("Training set size $N$")
    _ax.set_ylabel("time (ms)")
    _ax.set_yscale("log")
    _ax.set_title(_ttl)
    _ax.grid(True, which="both", alpha=0.3)
    _ax.legend(fontsize=8)
_plt.tight_layout()
_plt.savefig("plots/timing_all.png", dpi=150)
_plt.close()
print("Saved plots/timing_all.png")


# =============================================================================
# COMBINED 2-PANEL FIGURES FOR THE REPORT
# =============================================================================
print("\n" + "=" * 60)
print("  COMBINED 2-PANEL FIGURES")
print("=" * 60)

import matplotlib.pyplot as plt


def _plot_sweep_panel(ax, sweep, ylab, title, log_x=False):
    x = sweep['param_values']
    ax.errorbar(x, sweep['means'], yerr=sweep['stds'], marker='o', capsize=4)
    if log_x:
        ax.set_xscale('log')
    best = int(np.argmin(sweep['means']))
    ax.scatter([x[best]], [sweep['means'][best]], color='red', s=80, zorder=3,
               label=f"best: {sweep['param_name']}={x[best]}")
    ax.set_xlabel(sweep['param_name'])
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()


def _plot_ablation_panel(ax, ablation, ylab, title):
    names = [r['name'] for r in ablation]
    means = [r['mean'] for r in ablation]
    stds  = [r['std']  for r in ablation]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(ylab)
    ax.set_title(title)


# --- KNN: k-sweep classification + regression side by side ---
knn_sweep_cl = hyperparameter_sweep(
    KNN, 'k', [1, 3, 5, 7, 10, 15, 17, 20, 30, 50],
    fixed_kwargs={'task_kind': 'classification',
                  'metric': 'cosine', 'weighted': True},
    features=Xtr, labels=ytr_cl, task='classification', k=5,
)
knn_sweep_rg = hyperparameter_sweep(
    KNN, 'k', [1, 3, 5, 7, 10, 14, 15, 20, 30, 50],
    fixed_kwargs={'task_kind': 'regression',
                  'metric': 'cosine', 'weighted': True},
    features=Xtr, labels=ytr_rg, task='regression', k=5,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
_plot_sweep_panel(axes[0], knn_sweep_cl, "Error rate (1 - acc)",
                  "KNN — classification  (cosine, weighted)")
_plot_sweep_panel(axes[1], knn_sweep_rg, "MSE",
                  "KNN — regression  (cosine, weighted)")
plt.tight_layout()
plt.savefig("plots/knn_k_sweep.png", dpi=150)
plt.close()
print("Combined KNN k-sweep saved to plots/knn_k_sweep.png")


# --- LinReg ablation + LogReg ablation side by side ---
linreg_ablation = ablation_comparison(
    LinearRegression,
    configs=[
        ("vanilla",         {}),
        ("L2 λ=10 *",       {'lambda_reg': 10}),
        ("degree=2",        {'degree': 2}),
        ("degree=2 + int.", {'degree': 2, 'interaction': True}),
        ("degree=3",        {'degree': 3}),
    ],
    features=Xtr, labels=ytr_rg, task='regression', k=5,
)

logreg_ablation = ablation_comparison(
    LogisticRegression,
    configs=[
        ("vanilla GD",        {'lr': 0.3, 'max_iters': 1000}),
        ("xavier init",       {'lr': 0.3, 'max_iters': 1000, 'init': 'xavier'}),
        ("+ momentum β=0.9 *", {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9}),
        ("+ early stop",      {'lr': 0.3, 'max_iters': 1000, 'tol': 1e-6}),
        ("degree=2",          {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9,
                               'degree': 2}),
        ("degree=2 + int.",   {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9,
                               'degree': 2, 'interaction': True}),
        ("degree=3",          {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9,
                               'degree': 3}),
    ],
    features=Xtr, labels=ytr_cl, task='classification', k=5,
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
_plot_ablation_panel(axes[0], linreg_ablation, "MSE",
                     "Linear Regression ablation")
_plot_ablation_panel(axes[1], logreg_ablation, "Error rate (1 - acc)",
                     "Logistic Regression ablation")
plt.tight_layout()
plt.savefig("plots/linreg_logreg_ablation.png", dpi=150)
plt.close()
print("Combined LinReg+LogReg ablation saved to plots/linreg_logreg_ablation.png")


# --- LinReg λ sweep + LogReg lr sweep side by side (both log x-axis) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
_plot_sweep_panel(axes[0], linreg_lambda_sweep, "MSE",
                  "Linear Regression — λ sweep", log_x=True)
_plot_sweep_panel(axes[1], logreg_lr_sweep, "Error rate (1 - acc)",
                  "Logistic Regression — lr sweep", log_x=True)
plt.tight_layout()
plt.savefig("plots/linreg_logreg_sweep.png", dpi=150)
plt.close()
print("Combined LinReg λ + LogReg lr sweep saved to plots/linreg_logreg_sweep.png")


# --- All hyperparameter sweeps in a single 2x2 panel (for the report) ---
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
plt.close()
print("Combined 2x2 sweep panel saved to plots/all_sweeps.png")


# --- Kernelization experiments (Lecture 7) ----------------------------
from src.methods.kernel_ridge import (
    KernelRidgeRegression, KernelLeastSquaresClassifier,
)
from src.evaluation import stratified_kfold_cross_validation

print("\n--- Kernel Ridge Regression: kernel sweep (5-fold CV) ---")
krr_configs = [
    {'lambda_reg': 1.0, 'kernel': 'linear'},
    {'lambda_reg': 1.0, 'kernel': 'poly', 'degree': 2, 'coef0': 1.0},
    {'lambda_reg': 1.0, 'kernel': 'poly', 'degree': 3, 'coef0': 1.0},
    {'lambda_reg': 1.0, 'kernel': 'rbf',  'sigma': 3.0},
    {'lambda_reg': 1.0, 'kernel': 'rbf',  'sigma': 5.0},
    {'lambda_reg': 1.0, 'kernel': 'rbf',  'sigma': 10.0},
]
from src.optimization import grid_search  # just reuse existing helper
# use kfold_cross_validation directly for KRR
from src.evaluation import kfold_cross_validation
kfold_cross_validation(KernelRidgeRegression, krr_configs,
                       Xtr, ytr_rg, k=5, task='regression')

print("\n--- Kernel Least-Squares Classifier: kernel sweep (strat. 5-fold CV) ---")
kls_configs = [
    {'lambda_reg': 1.0, 'kernel': 'linear'},
    {'lambda_reg': 1.0, 'kernel': 'poly', 'degree': 2, 'coef0': 1.0},
    {'lambda_reg': 1.0, 'kernel': 'rbf',  'sigma': 3.0},
    {'lambda_reg': 1.0, 'kernel': 'rbf',  'sigma': 5.0},
    {'lambda_reg': 1.0, 'kernel': 'rbf',  'sigma': 10.0},
]
stratified_kfold_cross_validation(KernelLeastSquaresClassifier, kls_configs,
                                  Xtr, ytr_cl, k=5)


# --- LogReg L2 regularization + class-weight ablation ----------------
print("\n--- LogReg: L2 sweep + class weighting ---")
logreg_l2 = [
    {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9, 'lambda_reg': 0},
    {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9, 'lambda_reg': 0.1},
    {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9, 'lambda_reg': 1},
    {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9, 'lambda_reg': 10},
    {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9, 'lambda_reg': 100},
    {'lr': 0.3, 'max_iters': 1000, 'beta': 0.9, 'class_weight': 'balanced'},
]
stratified_kfold_cross_validation(LogisticRegression, logreg_l2,
                                  Xtr, ytr_cl, k=5)


# --- LogReg optimizer comparison: full-batch / SGD / mini-batch / LR-decay ---
print("\n--- LogReg: optimizer variants ---")
logreg_opt = [
    {'lr': 0.3,  'max_iters': 1000, 'beta': 0.9},
    {'lr': 0.05, 'max_iters': 2000, 'beta': 0.9, 'batch_size': 1},
    {'lr': 0.1,  'max_iters': 2000, 'beta': 0.9, 'batch_size': 32},
    {'lr': 0.2,  'max_iters': 2000, 'beta': 0.9, 'batch_size': 128},
    {'lr': 0.5,  'max_iters': 1000, 'beta': 0.9, 'lr_decay': 0.005},
]
stratified_kfold_cross_validation(LogisticRegression, logreg_opt,
                                  Xtr, ytr_cl, k=5)


# --- Linear SVM vs Logistic Regression ---------------------------------
from src.methods.svm import LinearSVM
print("\n--- Linear SVM (Crammer-Singer hinge) vs LogReg ---")
svm_configs = [
    {'lr': 0.05, 'max_iters': 2000, 'lambda_reg': 1e-4, 'batch_size': 64, 'beta': 0.9},
    {'lr': 0.05, 'max_iters': 2000, 'lambda_reg': 1e-3, 'batch_size': 64, 'beta': 0.9},
    {'lr': 0.05, 'max_iters': 2000, 'lambda_reg': 1e-2, 'batch_size': 64, 'beta': 0.9},
    {'lr': 0.05, 'max_iters': 2000, 'lambda_reg': 1e-3, 'batch_size': 64, 'beta': 0.9,
     'class_weight': 'balanced'},
]
stratified_kfold_cross_validation(LinearSVM, svm_configs, Xtr, ytr_cl, k=5)


# --- ROC curves (one-vs-rest) for best LogReg on a stratified hold-out ---
from src.evaluation import one_vs_rest_roc, plot_roc_curves

def _strat_split(y, frac, rng):
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        idx.append(ci[:int(frac * len(ci))])
    tr = np.concatenate(idx)
    va = np.setdiff1d(np.arange(len(y)), tr)
    return tr, va

rng = np.random.RandomState(7)
tr, va = _strat_split(ytr_cl, 0.8, rng)
np.random.seed(0)
lr_best = LogisticRegression(lr=0.3, max_iters=1000, beta=0.9)
lr_best.fit(Xtr[tr], ytr_cl[tr])
roc = one_vs_rest_roc(lr_best.predict_proba(Xtr[va]), ytr_cl[va],
                      class_names=['Low', 'Medium', 'High'])
plot_roc_curves(roc, title="LogReg one-vs-rest ROC",
                save_path="plots/logreg_roc.png")


print("\n" + "=" * 60)
print("  All report plots saved in ./plots/")
print("=" * 60)
