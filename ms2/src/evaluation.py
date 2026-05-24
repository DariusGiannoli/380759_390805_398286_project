"""Metrics, train/evaluate helpers, and k-fold cross-validation for MS2."""

import time
import numpy as np

from src.utils import (
    accuracy_fn, macrof1_fn, mse_fn,
    label_to_onehot, onehot_to_label, get_n_classes,
)
from src.methods.mlp import MLP
from src.methods.kmeans import KMeans
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, Identity, Softmax


# --------------------------------------------------------------------------
# A thin wrapper that gives the MLP the same fit(X, y) / predict(X) interface
# as the other methods, so all CV utilities below can be agnostic to method.
# --------------------------------------------------------------------------
class MLPWrapper:
    """Wraps an MLP with the (epochs, batch_size, lr, beta, hidden_dims,
    activation, loss, task) hyperparameters so we can pass it through k-fold
    CV like any other classifier/regressor."""

    def __init__(self, hidden_dims=(64, 32), activation='relu', loss='ce',
                 epochs=50, batch_size=32, lr=1e-2, beta=0.0,
                 task='classification', n_classes=None):
        self.hidden_dims = tuple(hidden_dims)
        self.activation_name = activation
        self.loss_name = loss
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.beta = float(beta)
        self.task = task
        self.n_classes = n_classes
        self.model = None

    def _build(self, D):
        hidden_act = ReLU if self.activation_name.lower() == 'relu' else Sigmoid
        if self.task == 'classification':
            C = self.n_classes
            dims = [D] + list(self.hidden_dims) + [C]
            acts = [hidden_act] * len(self.hidden_dims) + [Softmax]
        else:
            dims = [D] + list(self.hidden_dims) + [1]
            acts = [hidden_act] * len(self.hidden_dims) + [Identity]
        self.model = MLP(dimensions=tuple(dims), activations=tuple(acts))

    def fit(self, X, y):
        self._build(X.shape[1])
        if self.task == 'classification':
            if self.n_classes is None:
                self.n_classes = get_n_classes(y)
            y_train = label_to_onehot(y.astype(int), self.n_classes)
            loss = CrossEntropy if self.loss_name == 'ce' else MSE
        else:
            y_train = y.astype(np.float64).reshape(-1, 1)
            loss = MSE
        self.model.fit(
            X, y_train, loss=loss,
            epochs=self.epochs, batch_size=self.batch_size,
            learning_rate=self.lr, beta=self.beta,
        )
        return self.predict(X)

    def predict(self, X):
        out = self.model.predict(X)
        if self.task == 'classification':
            return onehot_to_label(out)
        return out.ravel()


# --------------------------------------------------------------------------
# Train + evaluate (single-shot)
# --------------------------------------------------------------------------
def train_and_evaluate_classification(method_obj, X_tr, y_tr, X_te, y_te):
    """Fit, predict, print and return accuracy + macro-F1 + timing."""
    t0 = time.time(); preds_tr = method_obj.fit(X_tr, y_tr); fit_t = time.time() - t0
    t0 = time.time(); preds_te = method_obj.predict(X_te); pred_t = time.time() - t0

    res = dict(
        acc_train = accuracy_fn(preds_tr, y_tr),
        f1_train  = macrof1_fn(preds_tr, y_tr),
        acc_test  = accuracy_fn(preds_te, y_te),
        f1_test   = macrof1_fn(preds_te, y_te),
        train_time=fit_t, pred_time=pred_t,
    )
    print(f"  Train : acc = {res['acc_train']:.2f}% | F1 = {res['f1_train']:.4f}")
    print(f"  Test  : acc = {res['acc_test']:.2f}% | F1 = {res['f1_test']:.4f}")
    print(f"  fit {fit_t*1e3:.0f} ms | predict {pred_t*1e3:.0f} ms")
    return res


def train_and_evaluate_regression(method_obj, X_tr, y_tr, X_te, y_te):
    """Fit, predict, print and return train/test MSE + timing."""
    t0 = time.time(); preds_tr = method_obj.fit(X_tr, y_tr); fit_t = time.time() - t0
    t0 = time.time(); preds_te = method_obj.predict(X_te); pred_t = time.time() - t0
    res = dict(
        train_mse=mse_fn(preds_tr, y_tr),
        test_mse =mse_fn(preds_te, y_te),
        train_time=fit_t, pred_time=pred_t,
    )
    print(f"  Train MSE = {res['train_mse']:.4f} | Test MSE = {res['test_mse']:.4f}")
    print(f"  fit {fit_t*1e3:.0f} ms | predict {pred_t*1e3:.0f} ms")
    return res


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
def _score(preds, labels, task):
    """Loss-like score (lower = better). Classification keeps accuracy error
    as a secondary diagnostic; model selection uses macro-F1 below."""
    if task == 'regression':
        return mse_fn(preds, labels)
    return 1.0 - accuracy_fn(preds, labels) / 100.0


def _f1_score(preds, labels):
    return macrof1_fn(preds, labels)


def kfold_cross_validation(method_class, kwargs_list, features, labels,
                            task, k=5, verbose=True):
    """Plain (random) k-fold CV for one or more configurations.

    Returns (best_kwargs, list_of_results) where each result has:
    'kwargs', 'mean', 'std', 'folds' (loss-like score per fold), and for
    classification 'f1_mean', 'f1_std', 'f1_folds'. Classification selects
    the best configuration by macro-F1; regression selects by MSE.
    """
    if verbose:
        print(f"  -- {k}-fold CV ({len(kwargs_list)} configs)")
    N = len(features)
    fold_size = N // k
    indices = np.arange(N)
    results = []
    for kwargs in kwargs_list:
        fold_scores, fold_f1 = [], []
        for i in range(k):
            val_idx = indices[i * fold_size : (i + 1) * fold_size]
            tr_idx  = np.concatenate([indices[:i * fold_size],
                                      indices[(i + 1) * fold_size:]])
            model = method_class(**kwargs)
            model.fit(features[tr_idx], labels[tr_idx])
            preds = model.predict(features[val_idx])
            fold_scores.append(_score(preds, labels[val_idx], task))
            if task == 'classification':
                fold_f1.append(_f1_score(preds, labels[val_idx]))
        out = dict(kwargs=kwargs,
                   mean=float(np.mean(fold_scores)),
                   std=float(np.std(fold_scores)),
                   folds=fold_scores)
        if task == 'classification':
            out['f1_mean']  = float(np.mean(fold_f1))
            out['f1_std']   = float(np.std(fold_f1))
            out['f1_folds'] = fold_f1
        results.append(out)
        if verbose:
            tag = f" | F1 {out['f1_mean']:.4f}" if task == 'classification' else ""
            print(f"     {kwargs} -> {out['mean']:.4f} +- {out['std']:.4f}{tag}")
    if task == 'classification':
        best = max(results, key=lambda r: r['f1_mean'])
    else:
        best = min(results, key=lambda r: r['mean'])
    if verbose:
        if task == 'classification':
            print(f"  -- best: {best['kwargs']} (F1 {best['f1_mean']:.4f})")
        else:
            print(f"  -- best: {best['kwargs']} (MSE {best['mean']:.4f})")
    return best['kwargs'], results


def stratified_kfold_cross_validation(method_class, kwargs_list, features,
                                       labels, k=5, verbose=True):
    """Stratified k-fold CV for classification: preserves class proportions
    in every fold so the rare High class is always present in val."""
    if verbose:
        print(f"  -- stratified {k}-fold CV ({len(kwargs_list)} configs)")
    classes = np.unique(labels)
    rs = np.random.RandomState(0)
    class_folds = {}
    for c in classes:
        cls_idx = np.where(labels == c)[0].copy()
        rs.shuffle(cls_idx)
        class_folds[c] = np.array_split(cls_idx, k)
    folds = [np.concatenate([class_folds[c][i] for c in classes]) for i in range(k)]
    results = []
    for kwargs in kwargs_list:
        fold_scores, fold_f1 = [], []
        for i in range(k):
            val_idx = folds[i]
            tr_idx  = np.concatenate([folds[j] for j in range(k) if j != i])
            model = method_class(**kwargs)
            model.fit(features[tr_idx], labels[tr_idx])
            preds = model.predict(features[val_idx])
            fold_scores.append(_score(preds, labels[val_idx], 'classification'))
            fold_f1.append(_f1_score(preds, labels[val_idx]))
        out = dict(kwargs=kwargs,
                   mean=float(np.mean(fold_scores)),
                   std=float(np.std(fold_scores)),
                   folds=fold_scores,
                   f1_mean=float(np.mean(fold_f1)),
                   f1_std=float(np.std(fold_f1)),
                   f1_folds=fold_f1)
        results.append(out)
        if verbose:
            print(f"     {kwargs} -> err {out['mean']:.4f} +- {out['std']:.4f}"
                  f" | F1 {out['f1_mean']:.4f}")
    best = max(results, key=lambda r: r['f1_mean'])
    if verbose:
        print(f"  -- best: {best['kwargs']} (F1 {best['f1_mean']:.4f})")
    return best['kwargs'], results
