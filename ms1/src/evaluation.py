"""Metrics, cross-validation, and the LOOCV shortcut for linear regression."""

import time
import numpy as np

from src.utils import mse_fn, accuracy_fn, macrof1_fn


def train_and_evaluate_classification(method_obj, train_features, train_labels,
                                       test_features, test_labels):
    """Fit a classification model, predict, report accuracy + F1 + timing."""
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
    """Fit a regression model, predict, report MSE + timing."""
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


def get_score(preds, labels, task):
    """Scalar score (lower is better): MSE for regression, error rate for classification."""
    if task == "regression":
        return mse_fn(preds, labels)
    elif task == "classification":
        return 1.0 - accuracy_fn(preds, labels) / 100.0
    raise ValueError(f"task must be 'regression' or 'classification', not {task!r}")


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


def stratified_kfold_cross_validation(method_class, method_kwargs_list,
                                       features, labels, k=5):
    """Stratified K-Fold CV (classification only) — preserves class proportions in every fold."""
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


def loocv_linear_regression(features, labels, lambda_reg):
    """
    Exact LOOCV for ridge linear regression using the hat matrix shortcut:
    LOO_error_i = residual_i / (1 - H_ii), where H = X(XᵀX + λI)⁻¹Xᵀ.
    Cost: one fit + O(N²).
    """
    N, D = features.shape
    X    = np.hstack([features, np.ones((N, 1))])

    reg         = np.eye(D + 1)
    reg[-1, -1] = 0.0
    XtX_reg     = X.T @ X + lambda_reg * reg

    H         = X @ np.linalg.solve(XtX_reg, X.T)
    w         = np.linalg.solve(XtX_reg, X.T @ labels)
    residuals = labels - X @ w

    h_diag  = np.diag(H)
    loo_err = residuals / (1 - h_diag)
    return float(np.mean(loo_err ** 2))
