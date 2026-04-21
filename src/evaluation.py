"""
Model evaluation infrastructure: metrics, CV, diagnostic plots.

This module is responsible for *scoring* a model.
For hyperparameter optimization (grid search, coarse-to-fine, etc.),
see optimization.py.
"""

import time
import numpy as np

from src.utils import mse_fn, accuracy_fn, macrof1_fn


# =============================================================================
# CORE TRAIN & EVALUATE
# =============================================================================

def train_and_evaluate_classification(method_obj, train_features, train_labels,
                                       test_features, test_labels):
    """
    Fits a classification model, predicts, reports accuracy + F1 + timing.

    Returns:
        dict with keys: acc_train, f1_train, acc_test, f1_test,
                        train_time, pred_time
    """
    t0 = time.time()
    preds_train = method_obj.fit(train_features, train_labels)
    train_time  = time.time() - t0

    t0 = time.time()
    preds_test  = method_obj.predict(test_features)
    pred_time   = time.time() - t0

    acc_train = accuracy_fn(preds_train, train_labels)
    f1_train  = macrof1_fn(preds_train, train_labels)
    acc_test  = accuracy_fn(preds_test,  test_labels)
    f1_test   = macrof1_fn(preds_test,   test_labels)

    print(f"\n{'='*52}")
    print(f"  Task : Classification")
    print(f"{'='*52}")
    print(f"  Train : accuracy = {acc_train:.3f}%  |  F1 = {f1_train:.6f}")
    print(f"  Test  : accuracy = {acc_test:.3f}%  |  F1 = {f1_test:.6f}")
    print(f"  Train time : {train_time:.4f}s  |  Predict time : {pred_time:.4f}s")

    return dict(acc_train=acc_train, f1_train=f1_train,
                acc_test=acc_test,   f1_test=f1_test,
                train_time=train_time, pred_time=pred_time)


def train_and_evaluate_regression(method_obj, train_features, train_labels,
                                   test_features, test_labels):
    """
    Fits a regression model, predicts, reports MSE + timing.

    Returns:
        dict with keys: train_mse, test_mse, train_time, pred_time
    """
    t0 = time.time()
    preds_train = method_obj.fit(train_features, train_labels)
    train_time  = time.time() - t0

    t0 = time.time()
    preds_test  = method_obj.predict(test_features)
    pred_time   = time.time() - t0

    train_mse = mse_fn(preds_train, train_labels)
    test_mse  = mse_fn(preds_test,  test_labels)

    print(f"\n{'='*52}")
    print(f"  Task : Regression")
    print(f"{'='*52}")
    print(f"  Train MSE : {train_mse:.6f}")
    print(f"  Test  MSE : {test_mse:.6f}")
    print(f"  Train time : {train_time:.4f}s  |  Predict time : {pred_time:.4f}s")

    return dict(train_mse=train_mse, test_mse=test_mse,
                train_time=train_time, pred_time=pred_time)


# =============================================================================
# SCORING HELPER
# =============================================================================

def get_score(preds, labels, task):
    """
    Returns a scalar score where LOWER is always better.
        Regression     : MSE
        Classification : error rate (1 - accuracy)
    """
    if task == "regression":
        return mse_fn(preds, labels)
    elif task == "classification":
        return 1.0 - accuracy_fn(preds, labels) / 100.0
    raise ValueError(f"Unknown task: {task}")


# =============================================================================
# K-FOLD CROSS-VALIDATION
# =============================================================================

def kfold_cross_validation(method_class, method_kwargs_list,
                            features, labels, task, k=5):
    """
    K-Fold CV for one or more configurations.

    Arguments:
        method_class       : class to instantiate
        method_kwargs_list : list of dicts, one per config
        k                  : number of folds

    Returns:
        best_kwargs (dict), results (list of dicts with 'kwargs', 'mean',
                                     'std', 'folds')
    """
    print(f"\n--- {k}-Fold CV ({len(method_kwargs_list)} configs) ---")

    N         = len(features)
    fold_size = N // k
    indices   = np.arange(N)
    results   = []

    for kwargs in method_kwargs_list:
        fold_scores = []

        for i in range(k):
            val_idx   = indices[i * fold_size : (i + 1) * fold_size]
            train_idx = np.concatenate([indices[:i * fold_size],
                                        indices[(i + 1) * fold_size:]])

            model = method_class(**kwargs)
            model.fit(features[train_idx], labels[train_idx])
            score = get_score(model.predict(features[val_idx]),
                              labels[val_idx], task)
            fold_scores.append(score)

        mean_s = np.mean(fold_scores)
        std_s  = np.std(fold_scores)
        results.append({'kwargs': kwargs, 'mean': mean_s,
                        'std': std_s, 'folds': fold_scores})
        print(f"  {kwargs}  →  {mean_s:.6f} ± {std_s:.6f}")

    best = min(results, key=lambda x: x['mean'])
    print(f"\n  Best : {best['kwargs']}  (mean = {best['mean']:.6f})")
    return best['kwargs'], results


# =============================================================================
# STRATIFIED K-FOLD (classification only)
# =============================================================================

def stratified_kfold_cross_validation(method_class, method_kwargs_list,
                                       features, labels, k=5):
    """
    Stratified K-Fold CV — preserves class proportions across folds.
    For classification only.
    """
    print(f"\n--- Stratified {k}-Fold CV ({len(method_kwargs_list)} configs) ---")

    classes = np.unique(labels)
    class_folds = {}
    for c in classes:
        cls_idx = np.where(labels == c)[0]
        np.random.RandomState(0).shuffle(cls_idx)
        class_folds[c] = np.array_split(cls_idx, k)

    folds = [np.concatenate([class_folds[c][i] for c in classes]) for i in range(k)]
    results = []

    for kwargs in method_kwargs_list:
        fold_scores = []

        for i in range(k):
            val_idx   = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            model = method_class(**kwargs)
            model.fit(features[train_idx], labels[train_idx])
            score = get_score(model.predict(features[val_idx]),
                              labels[val_idx], 'classification')
            fold_scores.append(score)

        mean_s, std_s = np.mean(fold_scores), np.std(fold_scores)
        results.append({'kwargs': kwargs, 'mean': mean_s,
                        'std': std_s, 'folds': fold_scores})
        print(f"  {kwargs}  →  {mean_s:.6f} ± {std_s:.6f}")

    best = min(results, key=lambda x: x['mean'])
    print(f"\n  Best : {best['kwargs']}  (mean = {best['mean']:.6f})")
    return best['kwargs'], results


# =============================================================================
# LOOCV SHORTCUT (linear regression only)
# =============================================================================

def loocv_linear_regression(features, labels, lambda_reg):
    """
    Exact LOOCV for linear regression using the hat matrix shortcut.
    Cost: one fit + O(N²).

    Formula:  LOO_error_i = residual_i / (1 - H_ii)
    where H = X(XᵀX + λI)⁻¹Xᵀ  is the hat matrix.
    """
    N, D = features.shape
    X    = np.hstack([features, np.ones((N, 1))])

    I         = np.eye(D + 1)
    I[-1, -1] = 0
    XtX_reg   = X.T @ X + lambda_reg * I

    H         = X @ np.linalg.solve(XtX_reg, X.T)
    w         = np.linalg.solve(XtX_reg, X.T @ labels)
    residuals = labels - X @ w

    h_diag  = np.diag(H)
    loo_err = residuals / (1 - h_diag)
    return float(np.mean(loo_err ** 2))


# =============================================================================
# DIAGNOSTIC PLOT: loss curve
# =============================================================================

def plot_loss_curve(loss_history, save_path="plots/loss_curve.png"):
    """Plots gradient descent convergence curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Gradient descent convergence")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


# =============================================================================
# CONFUSION MATRIX (classification)
# =============================================================================

def compute_confusion_matrix(pred_labels, true_labels, n_classes=None):
    """Rows = true class, columns = predicted class."""
    if n_classes is None:
        n_classes = int(max(pred_labels.max(), true_labels.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels.astype(int), pred_labels.astype(int)):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(pred_labels, true_labels, class_names=None,
                          save_path="plots/confusion_matrix.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    cm = compute_confusion_matrix(pred_labels, true_labels)
    C  = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    for i in range(C):
        for j in range(C):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    return cm


# =============================================================================
# LEARNING CURVES
# =============================================================================

def compute_learning_curve(method_class, method_kwargs, features, labels, task,
                           sizes=None, n_repeats=3):
    """
    Computes train/val scores as a function of training set size.
    Uses an internal 80/20 split, averaged across n_repeats shuffles.
    """
    N = len(features)
    val_size   = int(0.2 * N)
    train_pool = N - val_size

    if sizes is None:
        sizes = np.linspace(50, train_pool, 10, dtype=int)

    train_scores = {s: [] for s in sizes}
    val_scores   = {s: [] for s in sizes}

    for rep in range(n_repeats):
        rng = np.random.RandomState(rep)
        perm = rng.permutation(N)
        X_tr_all, y_tr_all = features[perm[:train_pool]], labels[perm[:train_pool]]
        X_val,    y_val    = features[perm[train_pool:]], labels[perm[train_pool:]]

        for s in sizes:
            X_sub, y_sub = X_tr_all[:s], y_tr_all[:s]
            model = method_class(**method_kwargs)
            model.fit(X_sub, y_sub)
            train_scores[s].append(get_score(model.predict(X_sub), y_sub, task))
            val_scores[s].append(get_score(model.predict(X_val), y_val, task))

    return dict(
        sizes=np.array(sizes),
        train_mean=np.array([np.mean(train_scores[s]) for s in sizes]),
        train_std =np.array([np.std (train_scores[s]) for s in sizes]),
        val_mean  =np.array([np.mean(val_scores[s])   for s in sizes]),
        val_std   =np.array([np.std (val_scores[s])   for s in sizes]),
    )


def plot_learning_curve(lc_results, task, title="Learning curve",
                        save_path="plots/learning_curve.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    sizes = lc_results['sizes']
    ylab  = "MSE" if task == "regression" else "Error rate (1 - acc)"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, lc_results['train_mean'], marker='o', label='Train')
    ax.fill_between(sizes,
                    lc_results['train_mean'] - lc_results['train_std'],
                    lc_results['train_mean'] + lc_results['train_std'], alpha=0.2)
    ax.plot(sizes, lc_results['val_mean'], marker='s', label='Validation')
    ax.fill_between(sizes,
                    lc_results['val_mean'] - lc_results['val_std'],
                    lc_results['val_mean'] + lc_results['val_std'], alpha=0.2)

    ax.set_xlabel("Training set size")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curve saved to {save_path}")


# =============================================================================
# TIMING MEASUREMENT
# =============================================================================

def timing_comparison(method_class, method_kwargs, features, labels,
                      sizes=None, n_repeats=3):
    """Measures train and predict time as a function of training set size."""
    N = len(features)
    if sizes is None:
        sizes = np.linspace(100, N, 8, dtype=int)

    train_times = {s: [] for s in sizes}
    pred_times  = {s: [] for s in sizes}

    for rep in range(n_repeats):
        rng  = np.random.RandomState(rep)
        perm = rng.permutation(N)

        for s in sizes:
            X_sub, y_sub = features[perm[:s]], labels[perm[:s]]
            model = method_class(**method_kwargs)

            t0 = time.time()
            model.fit(X_sub, y_sub)
            train_times[s].append(time.time() - t0)

            t0 = time.time()
            model.predict(X_sub)
            pred_times[s].append(time.time() - t0)

    return dict(
        sizes=np.array(sizes),
        train_mean=np.array([np.mean(train_times[s]) for s in sizes]),
        train_std =np.array([np.std (train_times[s]) for s in sizes]),
        pred_mean =np.array([np.mean(pred_times[s])  for s in sizes]),
        pred_std  =np.array([np.std (pred_times[s])  for s in sizes]),
    )


def plot_timing(timing_results, title="Timing scaling",
                save_path="plots/timing.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    s = timing_results['sizes']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(s, timing_results['train_mean'], yerr=timing_results['train_std'],
                marker='o', capsize=4, label='Train')
    ax.errorbar(s, timing_results['pred_mean'], yerr=timing_results['pred_std'],
                marker='s', capsize=4, label='Predict')
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Timing plot saved to {save_path}")
