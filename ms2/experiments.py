"""Generates every figure and number used in report.tex.

Outputs PNGs to figures/ and a JSON summary to figures/summary.json.
Run from inside ms2/ with:  python experiments.py
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import (
    normalize_fn, label_to_onehot, onehot_to_label,
    accuracy_fn, macrof1_fn, mse_fn, get_n_classes,
)
from src.methods.mlp import MLP
from src.methods.kmeans import KMeans
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, Identity, Softmax
from src.evaluation import (
    MLPWrapper,
    stratified_kfold_cross_validation,
    kfold_cross_validation,
)
from src.optimization import hyperparameter_sweep

# -- Style --------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 200,
    # Larger font sizes so axis labels and tick labels remain readable
    # when the PNG is scaled down to ~0.32 or 0.48 \columnwidth in LaTeX.
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {'mlp': '#1f77b4', 'kmeans': '#d62728', 'aux': '#2ca02c', 'gray': '#7f7f7f'}

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Load + prepare data (shuffle BEFORE split; addresses MS1 feedback)
# ---------------------------------------------------------------------
def load_data(path='data/features.npz', val_frac=0.2, seed=0):
    feat = np.load(path, allow_pickle=True)
    X_tr_all = feat['xtrain']
    X_te     = feat['xtest']
    y_tr_reg     = feat['ytrainreg']
    y_te_reg     = feat['ytestreg']
    y_tr_clf     = feat['ytrainclassif']
    y_te_clf     = feat['ytestclassif']

    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(X_tr_all))
    X_tr_all = X_tr_all[perm]
    y_tr_reg = y_tr_reg[perm]
    y_tr_clf = y_tr_clf[perm]

    val_n = int(val_frac * len(X_tr_all))
    X_val, X_tr = X_tr_all[-val_n:], X_tr_all[:-val_n]
    y_val_reg, y_tr_reg_s = y_tr_reg[-val_n:], y_tr_reg[:-val_n]
    y_val_clf, y_tr_clf_s = y_tr_clf[-val_n:], y_tr_clf[:-val_n]

    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True); std[std == 0] = 1.0
    X_tr  = normalize_fn(X_tr,  mean, std)
    X_val = normalize_fn(X_val, mean, std)
    X_te  = normalize_fn(X_te,  mean, std)

    return dict(
        X_tr=X_tr, y_tr_clf=y_tr_clf_s, y_tr_reg=y_tr_reg_s,
        X_val=X_val, y_val_clf=y_val_clf, y_val_reg=y_val_reg,
        X_te=X_te, y_te_clf=y_te_clf, y_te_reg=y_te_reg,
        mean=mean, std=std,
    )


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------
def mlp_lr_sweep(data, lrs, hidden=(64, 32), epochs=40, batch_size=32, beta=0.9):
    """Sweep learning rate via stratified 5-fold CV on the train split."""
    print("\n[MLP] learning rate sweep")
    np.random.seed(100)
    C = get_n_classes(np.concatenate([data['y_tr_clf'], data['y_val_clf']]))
    fixed = dict(hidden_dims=hidden, activation='relu', loss='ce',
                 epochs=epochs, batch_size=batch_size, beta=beta,
                 task='classification', n_classes=C)
    res = hyperparameter_sweep(
        MLPWrapper, 'lr', list(lrs), fixed,
        data['X_tr'], data['y_tr_clf'],
        task='classification', k=5, stratified=True, verbose=True,
    )
    return res


def mlp_arch_sweep(data, archs, lr, epochs=40, batch_size=32, beta=0.9):
    """CV over a few hidden-layer architectures at the selected lr."""
    print("\n[MLP] architecture sweep")
    np.random.seed(100)
    C = get_n_classes(np.concatenate([data['y_tr_clf'], data['y_val_clf']]))
    fixed = dict(activation='relu', loss='ce',
                 epochs=epochs, batch_size=batch_size, lr=lr, beta=beta,
                 task='classification', n_classes=C)
    res = hyperparameter_sweep(
        MLPWrapper, 'hidden_dims', list(archs), fixed,
        data['X_tr'], data['y_tr_clf'],
        task='classification', k=5, stratified=True, verbose=True,
    )
    return res


def mlp_ablation(data, best_lr, best_arch, epochs=40, batch_size=32, beta=0.9):
    """Compact 5-fold CV ablation around the selected config: ReLU vs
    Sigmoid, CE vs MSE-on-onehot, momentum, batch size, epochs."""
    print("\n[MLP] ablation (5-fold stratified CV)")
    np.random.seed(100)
    C = get_n_classes(np.concatenate([data['y_tr_clf'], data['y_val_clf']]))
    base = dict(hidden_dims=best_arch, activation='relu', loss='ce',
                epochs=epochs, batch_size=batch_size, lr=best_lr, beta=beta,
                task='classification', n_classes=C)

    configs = {
        'relu':     {},
        'sigmoid':  dict(activation='sigmoid'),
        'ce':       {},
        'mse_oh':   dict(loss='mse'),
        'beta_0':   dict(beta=0.0),
        'beta_0.5': dict(beta=0.5),
        'beta_0.9': {},
        'bs_8':     dict(batch_size=8),
        'bs_128':   dict(batch_size=128),
        'ep_10':    dict(epochs=10),
        'ep_100':   dict(epochs=100),
    }
    out = {}
    for name, override in configs.items():
        cfg = dict(base, **override)
        _, results = stratified_kfold_cross_validation(
            MLPWrapper, [cfg], data['X_tr'], data['y_tr_clf'],
            k=5, verbose=False,
        )
        out[name] = dict(
            f1_mean=float(results[0]['f1_mean']),
            f1_std =float(results[0]['f1_std']),
        )
        print(f"  {name:10s}: F1 {out[name]['f1_mean']:.4f}"
              f" +- {out[name]['f1_std']:.4f}")
    return out


def mlp_reg_lr_sweep(data, lrs, hidden=(32,), epochs=40, batch_size=32, beta=0.9):
    """Sweep learning rate for MLP regression via 5-fold CV."""
    print("\n[MLP] regression learning rate sweep")
    np.random.seed(100)
    fixed = dict(hidden_dims=hidden, activation='relu', loss='mse',
                 epochs=epochs, batch_size=batch_size, beta=beta,
                 task='regression')
    res = hyperparameter_sweep(
        MLPWrapper, 'lr', list(lrs), fixed,
        data['X_tr'], data['y_tr_reg'],
        task='regression', k=5, stratified=False, verbose=True,
    )
    return res


def kmeans_k_sweep(data, Ks, n_restarts=10):
    """Sweep K via stratified 5-fold CV + record inertia for elbow."""
    print("\n[KMeans] K sweep + inertia")
    np.random.seed(100)
    fixed = dict(init='kmeans++', n_restarts=n_restarts, max_iters=200)
    res = hyperparameter_sweep(
        KMeans, 'K', list(Ks), fixed,
        data['X_tr'], data['y_tr_clf'],
        task='classification', k=5, stratified=True, verbose=True,
    )
    # Inertia on the train split (single fit per K with same n_restarts).
    inertias = []
    for K in Ks:
        m = KMeans(K=K, max_iters=200, init='kmeans++', n_restarts=n_restarts)
        m.fit(data['X_tr'], data['y_tr_clf'])
        inertias.append(m.inertia_)
    res['inertias'] = np.array(inertias)
    return res


def kmeans_init_compare(data, K, n_runs=20):
    """Compare init strategies at the selected K. Reports mean +- std of
    final inertia and val macro-F1 over n_runs random seeds."""
    print(f"\n[KMeans] init strategy comparison at K={K}")
    results = {}
    for name, kwargs in [
        ('random',       dict(init='random',   n_restarts=1)),
        ('kmeans++',     dict(init='kmeans++', n_restarts=1)),
        ('best-of-10',   dict(init='random',   n_restarts=10)),
        ('km++ x10',     dict(init='kmeans++', n_restarts=10)),
    ]:
        inertias, f1s = [], []
        for seed in range(n_runs):
            np.random.seed(seed)
            m = KMeans(K=K, max_iters=200, **kwargs)
            preds_tr = m.fit(data['X_tr'], data['y_tr_clf'])
            preds_val = m.predict(data['X_val'])
            inertias.append(m.inertia_)
            f1s.append(macrof1_fn(preds_val, data['y_val_clf']))
        results[name] = dict(
            inertia_mean=float(np.mean(inertias)),
            inertia_std =float(np.std(inertias)),
            f1_mean     =float(np.mean(f1s)),
            f1_std      =float(np.std(f1s)),
        )
        print(f"  {name:10s}: inertia {results[name]['inertia_mean']:.1f}"
              f" +- {results[name]['inertia_std']:.1f}"
              f" | F1 {results[name]['f1_mean']:.4f}"
              f" +- {results[name]['f1_std']:.4f}")
    return results


def mlp_loss_curve(data, lr, hidden, epochs, batch_size, beta):
    """Fit one MLP with track_loss=True for the training curve plot."""
    print("\n[MLP] training loss curve")
    C = get_n_classes(np.concatenate([data['y_tr_clf'], data['y_val_clf']]))
    np.random.seed(0)
    dims = [data['X_tr'].shape[1]] + list(hidden) + [C]
    acts = [ReLU] * len(hidden) + [Softmax]
    m = MLP(dimensions=tuple(dims), activations=tuple(acts))
    y_one_hot = label_to_onehot(data['y_tr_clf'].astype(int), C)
    m.fit(data['X_tr'], y_one_hot, loss=CrossEntropy,
          epochs=epochs, batch_size=batch_size, learning_rate=lr,
          beta=beta, track_loss=True)
    train_curve = list(m.loss_history)
    # Compute val loss at the same epochs we tracked
    val_pred = m.predict(data['X_val'])
    val_y_oh = label_to_onehot(data['y_val_clf'].astype(int), C)
    val_loss = CrossEntropy.loss(val_y_oh, val_pred)
    return train_curve, val_loss


def _per_class_prf(y_true, y_pred, n_classes=3):
    """Per-class precision, recall, F1. Returns dict {class_idx: (p, r, f1)}."""
    out = {}
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[c] = (float(p), float(r), float(f1))
    return out


def final_test(data, mlp_cfg, km_cfg, mlp_reg_cfg=None):
    """Refit on train+val, evaluate on the official test set."""
    if mlp_reg_cfg is None:
        mlp_reg_cfg = mlp_cfg
    print("\n[final] refit on train+val and evaluate on test")
    X_full = np.concatenate([data['X_tr'], data['X_val']])
    y_clf_full = np.concatenate([data['y_tr_clf'], data['y_val_clf']])
    y_reg_full = np.concatenate([data['y_tr_reg'], data['y_val_reg']])
    C = get_n_classes(y_clf_full)
    out = {}

    # MLP classification
    np.random.seed(100)
    dims = [X_full.shape[1]] + list(mlp_cfg['hidden']) + [C]
    acts = [ReLU] * len(mlp_cfg['hidden']) + [Softmax]
    m = MLP(dimensions=tuple(dims), activations=tuple(acts))
    y_oh = label_to_onehot(y_clf_full.astype(int), C)
    t0 = time.time()
    m.fit(X_full, y_oh, loss=CrossEntropy,
          epochs=mlp_cfg['epochs'], batch_size=mlp_cfg['batch_size'],
          learning_rate=mlp_cfg['lr'], beta=mlp_cfg['beta'])
    fit_t = time.time() - t0
    preds = onehot_to_label(m.predict(data['X_te']))
    out['mlp_clf'] = dict(
        test_acc=float(accuracy_fn(preds, data['y_te_clf'])),
        test_f1 =float(macrof1_fn(preds, data['y_te_clf'])),
        fit_time=fit_t,
        config  =mlp_cfg,
        preds   =preds,
        per_class=_per_class_prf(data['y_te_clf'], preds),
    )
    print(f"  MLP clf: acc {out['mlp_clf']['test_acc']:.2f}%"
          f" | F1 {out['mlp_clf']['test_f1']:.4f}"
          f" | fit {fit_t*1e3:.0f} ms")

    # MLP regression — same architecture but Identity output, MSE loss.
    # Uses its own (typically smaller) learning rate from mlp_reg_cfg.
    np.random.seed(100)
    dims = [X_full.shape[1]] + list(mlp_reg_cfg['hidden']) + [1]
    acts = [ReLU] * len(mlp_reg_cfg['hidden']) + [Identity]
    m = MLP(dimensions=tuple(dims), activations=tuple(acts))
    y_col = y_reg_full.astype(np.float64).reshape(-1, 1)
    t0 = time.time()
    m.fit(X_full, y_col, loss=MSE,
          epochs=mlp_reg_cfg['epochs'], batch_size=mlp_reg_cfg['batch_size'],
          learning_rate=mlp_reg_cfg['lr'], beta=mlp_reg_cfg['beta'])
    fit_t = time.time() - t0
    preds = m.predict(data['X_te']).ravel()
    out['mlp_reg'] = dict(
        test_mse=float(mse_fn(preds, data['y_te_reg'])),
        fit_time=fit_t,
        config  =mlp_reg_cfg,
    )
    print(f"  MLP reg: MSE {out['mlp_reg']['test_mse']:.4f}"
          f" | fit {fit_t*1e3:.0f} ms")

    # K-Means classification
    np.random.seed(100)
    km = KMeans(K=km_cfg['K'], max_iters=200,
                init=km_cfg['init'], n_restarts=km_cfg['n_restarts'])
    t0 = time.time()
    km.fit(X_full, y_clf_full)
    fit_t = time.time() - t0
    preds = km.predict(data['X_te'])
    out['km_clf'] = dict(
        test_acc=float(accuracy_fn(preds, data['y_te_clf'])),
        test_f1 =float(macrof1_fn(preds, data['y_te_clf'])),
        fit_time=fit_t,
        config  =km_cfg,
        preds   =preds,
        per_class=_per_class_prf(data['y_te_clf'], preds),
    )
    print(f"  KM  clf: acc {out['km_clf']['test_acc']:.2f}%"
          f" | F1 {out['km_clf']['test_f1']:.4f}"
          f" | fit {fit_t*1e3:.0f} ms")
    return out


# ---------------------------------------------------------------------
# Figures (each panel saved as its own PNG so report.tex can use the
# LaTeX subfigure environment for cleaner layout).
# ---------------------------------------------------------------------
PANEL_SIZE = (3.4, 2.6)  # inches per single panel; ~4:3 looks good in 2-col


def _save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def fig_mlp_lr(lr_res, save):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.errorbar(lr_res['param_values'], lr_res['f1_means'],
                yerr=lr_res['f1_stds'], marker='o', color=COLORS['mlp'],
                capsize=2, lw=1.2, ms=4)
    ax.set_xscale('log')
    ax.set_xlabel('learning rate $\\eta$')
    ax.set_ylabel('5-fold CV macro-F1')
    best_lr_idx = int(np.argmax(lr_res['f1_means']))
    ax.scatter([lr_res['param_values'][best_lr_idx]],
               [lr_res['f1_means'][best_lr_idx]],
               marker='*', s=110, color='red', zorder=5,
               label='selected')
    ax.legend(loc='lower right', frameon=False)
    _save(fig, save)


def fig_mlp_arch(arch_res, save):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    arch_labels = [str(a) for a in arch_res['param_values']]
    means = arch_res['f1_means']
    stds  = arch_res['f1_stds']
    x = np.arange(len(arch_labels))
    ax.bar(x, means, yerr=stds, color=COLORS['mlp'], alpha=0.85,
           capsize=2, error_kw=dict(lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=25, ha='right')
    ax.set_ylabel('5-fold CV macro-F1')
    ax.set_xlabel('hidden layer dimensions')
    ax.set_ylim(min(means) - 0.04, max(means) + 0.02)
    _save(fig, save)


def fig_mlp_loss(loss_curve, save):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    train_curve, val_loss = loss_curve
    epochs = np.arange(1, len(train_curve) + 1)
    ax.plot(epochs, train_curve, color=COLORS['mlp'], lw=1.5, label='train')
    ax.axhline(val_loss, color=COLORS['gray'], ls='--', lw=1.2,
               label='val (final)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('cross-entropy loss')
    ax.legend(loc='upper right', frameon=False)
    _save(fig, save)


def fig_kmeans_k(k_res, save):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    Ks = np.array(k_res['param_values'])
    f1 = k_res['f1_means']
    f1_std = k_res['f1_stds']
    inertia = k_res['inertias']
    inertia_norm = (inertia - inertia.min()) / (inertia.max() - inertia.min())

    ax.errorbar(Ks, f1, yerr=f1_std, marker='o', color=COLORS['kmeans'],
                capsize=2, lw=1.2, ms=4)
    ax.set_xlabel('number of clusters $K$')
    ax.set_ylabel('5-fold CV macro-F1', color=COLORS['kmeans'])
    ax.tick_params(axis='y', labelcolor=COLORS['kmeans'])

    ax2 = ax.twinx()
    ax2.plot(Ks, inertia_norm, color=COLORS['gray'], lw=1.2, marker='s',
             ms=3)
    ax2.set_ylabel('inertia (min–max norm.)', color=COLORS['gray'])
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])
    ax2.spines['top'].set_visible(False)

    best_k = Ks[np.argmax(f1)]
    ax.axvline(best_k, color='red', ls=':', lw=0.8)
    ax.text(best_k, ax.get_ylim()[0], f' K*={best_k}', color='red',
            va='bottom', fontsize=8)
    _save(fig, save)


def fig_kmeans_init(init_cmp, save):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    names = list(init_cmp.keys())
    f1m = [init_cmp[n]['f1_mean'] for n in names]
    f1s = [init_cmp[n]['f1_std']  for n in names]
    x = np.arange(len(names))
    ax.bar(x, f1m, yerr=f1s, color=COLORS['kmeans'], alpha=0.85,
           capsize=2, error_kw=dict(lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel('val macro-F1 (20 seeds)')
    ax.set_xlabel('initialisation strategy')
    _save(fig, save)


def fig_comparison_clf(ms1_clf, ms2_clf, save):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    names = list(ms1_clf.keys()) + list(ms2_clf.keys())
    vals  = list(ms1_clf.values()) + list(ms2_clf.values())
    colors = [COLORS['gray']] * len(ms1_clf) \
        + [COLORS['mlp']] * (len(ms2_clf) - 1) + [COLORS['kmeans']]
    x = np.arange(len(names))
    ax.barh(x, vals, color=colors, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel('test macro-F1 (higher = better)')
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18)
    _save(fig, save)


def _row_normalised_cm(y_true, y_pred, n_classes):
    """Confusion matrix with rows normalised to sum to 1 (per-class recall view).
    Returns (counts, normalised) as (n_classes, n_classes) arrays."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    return cm, cm_norm


def fig_confusion_matrix(y_true, y_pred, title, save, n_classes=3,
                         labels=('Low', 'Medium', 'High'), cmap='Blues'):
    cm, cm_norm = _row_normalised_cm(y_true, y_pred, n_classes)
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0.0, vmax=1.0, aspect='equal')
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    # Cell annotations: recall (count)
    for i in range(n_classes):
        for j in range(n_classes):
            colour = 'white' if cm_norm[i, j] > 0.55 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.2f}\n({cm[i,j]})',
                    ha='center', va='center', color=colour, fontsize=10)
    _save(fig, save)


def fig_comparison_reg(ms1_reg, ms2_reg, save):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    names = list(ms1_reg.keys()) + list(ms2_reg.keys())
    vals  = list(ms1_reg.values()) + list(ms2_reg.values())
    colors = [COLORS['gray']] * len(ms1_reg) + [COLORS['mlp']] * len(ms2_reg)
    x = np.arange(len(names))
    ax.barh(x, vals, color=colors, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel('test MSE (lower = better)')
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18)
    _save(fig, save)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main():
    np.random.seed(100)
    data = load_data()

    # ----- MLP sweeps -----
    lr_res = mlp_lr_sweep(
        data, lrs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        hidden=(64, 32), epochs=40, batch_size=32, beta=0.9,
    )
    best_lr_idx = int(np.argmax(lr_res['f1_means']))
    best_lr = lr_res['param_values'][best_lr_idx]
    print(f"  >> best lr = {best_lr}  (CV F1 = {lr_res['f1_means'][best_lr_idx]:.4f})")

    arch_res = mlp_arch_sweep(
        data,
        archs=[(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
        lr=best_lr, epochs=40, batch_size=32, beta=0.9,
    )
    best_arch_idx = int(np.argmax(arch_res['f1_means']))
    best_arch = arch_res['param_values'][best_arch_idx]
    print(f"  >> best arch = {best_arch}  (CV F1 = {arch_res['f1_means'][best_arch_idx]:.4f})")

    loss_curve = mlp_loss_curve(
        data, lr=best_lr, hidden=best_arch,
        epochs=40, batch_size=32, beta=0.9,
    )

    # Compact ablation around the selected config (ReLU vs Sigmoid,
    # CE vs MSE, momentum, batch size, epochs).
    ablation = mlp_ablation(
        data, best_lr=best_lr, best_arch=best_arch,
        epochs=40, batch_size=32, beta=0.9,
    )

    # MLP regression: sweep the learning rate on the same log grid.
    reg_lr_res = mlp_reg_lr_sweep(
        data, lrs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        hidden=best_arch, epochs=40, batch_size=32, beta=0.9,
    )
    # NaN-safe argmin (lr=0.1 can diverge for the regression MLP).
    reg_means = np.nan_to_num(np.asarray(reg_lr_res['means'], dtype=float),
                              nan=np.inf, posinf=np.inf)
    best_reg_lr_idx = int(np.argmin(reg_means))
    best_reg_lr = reg_lr_res['param_values'][best_reg_lr_idx]
    print(f"  >> best regression lr = {best_reg_lr}"
          f"  (CV MSE = {reg_means[best_reg_lr_idx]:.4f})")

    # ----- K-Means sweeps -----
    k_res = kmeans_k_sweep(data, Ks=[3, 5, 10, 15, 20, 30, 50], n_restarts=10)
    best_k_idx = int(np.argmax(k_res['f1_means']))
    best_K = k_res['param_values'][best_k_idx]
    print(f"  >> best K = {best_K}  (CV F1 = {k_res['f1_means'][best_k_idx]:.4f})")

    init_cmp = kmeans_init_compare(data, K=best_K, n_runs=20)

    # ----- Final test-set evaluation -----
    mlp_cfg     = dict(hidden=best_arch, lr=best_lr,     beta=0.9,
                       epochs=40, batch_size=32)
    mlp_reg_cfg = dict(hidden=best_arch, lr=best_reg_lr, beta=0.9,
                       epochs=40, batch_size=32)
    km_cfg      = dict(K=best_K, init='kmeans++', n_restarts=10)
    final = final_test(data, mlp_cfg, km_cfg, mlp_reg_cfg=mlp_reg_cfg)

    # ----- Figures (one PNG per panel for cleaner LaTeX subfigure layout) -----
    fig_mlp_lr  (lr_res,    save=os.path.join(FIG_DIR, 'mlp_lr.png'))
    fig_mlp_arch(arch_res,  save=os.path.join(FIG_DIR, 'mlp_arch.png'))
    fig_mlp_loss(loss_curve,save=os.path.join(FIG_DIR, 'mlp_loss.png'))
    fig_kmeans_k   (k_res,    save=os.path.join(FIG_DIR, 'kmeans_k.png'))
    fig_kmeans_init(init_cmp, save=os.path.join(FIG_DIR, 'kmeans_init.png'))

    # MS1 numbers come from the published report; MS2 numbers from final_test.
    ms1_clf = {'LogReg (MS1)': 0.706, 'SVM (MS1)': 0.733,
               'KNN (MS1)':    0.623}
    ms2_clf = {'MLP (MS2)':    final['mlp_clf']['test_f1'],
               'K-Means (MS2)':final['km_clf']['test_f1']}
    ms1_reg = {'LinReg (MS1)': 0.993, 'KRR lin (MS1)': 0.969,
               'KNN (MS1)':    1.62}
    ms2_reg = {'MLP (MS2)':    final['mlp_reg']['test_mse']}
    fig_comparison_clf(ms1_clf, ms2_clf,
                       save=os.path.join(FIG_DIR, 'comparison_clf.png'))
    fig_comparison_reg(ms1_reg, ms2_reg,
                       save=os.path.join(FIG_DIR, 'comparison_reg.png'))

    # Confusion matrices for the two MS2 classifiers (test set).
    fig_confusion_matrix(
        data['y_te_clf'], final['mlp_clf']['preds'],
        title='MLP (MS2) — test set',
        save=os.path.join(FIG_DIR, 'cm_mlp.png'),
    )
    fig_confusion_matrix(
        data['y_te_clf'], final['km_clf']['preds'],
        title='K-Means (MS2) — test set',
        save=os.path.join(FIG_DIR, 'cm_kmeans.png'),
        cmap='Reds',
    )

    # Strip the (non-JSON-serializable) prediction arrays before dumping.
    final['mlp_clf'].pop('preds', None)
    final['km_clf'].pop('preds', None)

    # ----- JSON dump (for report copy/paste) -----
    summary = dict(
        mlp_lr_sweep = dict(
            lrs=list(lr_res['param_values']),
            f1=list(map(float, lr_res['f1_means'])),
            f1_std=list(map(float, lr_res['f1_stds'])),
            best_lr=best_lr,
        ),
        mlp_arch_sweep = dict(
            archs=[str(a) for a in arch_res['param_values']],
            f1=list(map(float, arch_res['f1_means'])),
            best_arch=str(best_arch),
        ),
        kmeans_k_sweep = dict(
            Ks=list(k_res['param_values']),
            f1=list(map(float, k_res['f1_means'])),
            inertias=list(map(float, k_res['inertias'])),
            best_K=int(best_K),
        ),
        kmeans_init_compare = init_cmp,
        mlp_ablation = ablation,
        mlp_reg_lr_sweep = dict(
            lrs=list(reg_lr_res['param_values']),
            mse=list(map(float, reg_lr_res['means'])),
            mse_std=list(map(float, reg_lr_res['stds'])),
            best_lr=best_reg_lr,
        ),
        final = final,
    )
    with open(os.path.join(FIG_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {os.path.join(FIG_DIR, 'summary.json')}")
    return summary


if __name__ == '__main__':
    main()
