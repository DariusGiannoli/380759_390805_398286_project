"""
Generate a 1x2 confusion-matrix figure for Linear SVM and class-weighted
Logistic Regression on the held-out test set.

Row-normalized (each row sums to 1.0) so each cell shows per-class
*recall*; raw counts are displayed in parentheses underneath. Both
panels share the same [0, 1] colour scale so the reader can compare
recall at a glance.

Saves plots/confusion_matrices.png at dpi=200.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.methods.logistic_regression import LogisticRegression
from src.methods.extension.svm import LinearSVM
from src.utils import normalize_fn


CLASS_NAMES = ['Low', 'Medium', 'High']
N_CLASSES   = 3


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_split():
    """Stratified 80/20 split; z-scored on the training fold only."""
    d = np.load('data/features.npz')
    X_raw, y = d['xtrain'], d['ytrainclassif'].astype(int)

    rng = np.random.RandomState(7)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0].copy()
        rng.shuffle(ci)
        idx.append(ci[:int(0.8 * len(ci))])
    tr = np.concatenate(idx)
    te = np.setdiff1d(np.arange(len(y)), tr)

    mu, sd = X_raw[tr].mean(0), X_raw[tr].std(0)
    sd[sd == 0] = 1
    return (normalize_fn(X_raw[tr], mu, sd), y[tr],
            normalize_fn(X_raw[te], mu, sd), y[te])


# --------------------------------------------------------------------------- #
# Confusion matrix helpers
# --------------------------------------------------------------------------- #

def confusion_matrix(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def _plot_one(ax, cm, title, show_ylabel=True):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm / np.maximum(row_sums, 1)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('True', fontsize=9)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    # Cell annotations: recall + raw count
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            cnt = int(cm[i, j])
            color = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}\n({cnt})',
                    ha='center', va='center',
                    color=color, fontsize=8)
    return im


def plot_confusion_matrices(y_true, y_pred_svm, y_pred_logreg,
                            save_path='plots/confusion_matrices.png'):
    """Side-by-side 1x2 confusion matrices with a shared [0,1] colour scale."""
    cm_svm    = confusion_matrix(y_true, y_pred_svm)
    cm_logreg = confusion_matrix(y_true, y_pred_logreg)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7, 3.2),
                                      facecolor='white')
    _plot_one(ax_l, cm_svm,    'Linear SVM',            show_ylabel=True)
    im = _plot_one(ax_r, cm_logreg, 'Class-weighted LogReg', show_ylabel=False)

    fig.subplots_adjust(wspace=0.3)
    # Shared colorbar on the right
    fig.colorbar(im, ax=[ax_l, ax_r], shrink=0.85,
                 label='recall', fraction=0.05, pad=0.02)

    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {save_path}')
    return cm_svm, cm_logreg


# --------------------------------------------------------------------------- #
# Generate predictions from trained models
# --------------------------------------------------------------------------- #

def main():
    Xtr, ytr, Xte, yte = load_split()

    np.random.seed(0)
    svm = LinearSVM(lr=0.05, max_iters=2000, lambda_reg=1e-2,
                    batch_size=64, beta=0.9)
    svm.fit(Xtr, ytr)
    y_pred_svm = svm.predict(Xte)

    np.random.seed(0)
    logreg_cw = LogisticRegression(lr=0.3, max_iters=1000, beta=0.9,
                                   class_weight='balanced')
    logreg_cw.fit(Xtr, ytr)
    y_pred_logreg = logreg_cw.predict(Xte)

    cm_svm, cm_logreg = plot_confusion_matrices(yte, y_pred_svm, y_pred_logreg)

    # Per-class recall printout (useful for the report text)
    for name, cm in [('Linear SVM', cm_svm),
                     ('Class-weighted LogReg', cm_logreg)]:
        print(f'\n{name} — per-class recall:')
        for c, cname in enumerate(CLASS_NAMES):
            total = cm[c].sum()
            recall = cm[c, c] / max(total, 1)
            print(f'  {cname:<7} {recall:.3f}  ({cm[c, c]}/{total})')


if __name__ == '__main__':
    main()
