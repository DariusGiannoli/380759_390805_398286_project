import numpy as np
import time
from src.utils import mse_fn, accuracy_fn, macrof1_fn

# =============================================================================
# CORE TRAIN & EVALUATE
# =============================================================================

def train_and_evaluate_classification(method_obj, train_features, train_labels,
                                       test_features, test_labels):
    """
    Fits a classification model, predicts, reports accuracy + F1 + timing.

    Arguments:
        method_obj: any object with fit() and predict()
        train_features, train_labels: training data
        test_features, test_labels: test/validation data
    Returns:
        dict with keys: acc_train, f1_train, acc_test, f1_test,
                        train_time, pred_time
    """
    # Train
    t0 = time.time()
    preds_train = method_obj.fit(train_features, train_labels)
    train_time  = time.time() - t0

    # Predict
    t0 = time.time()
    preds_test  = method_obj.predict(test_features)
    pred_time   = time.time() - t0

    # Metrics
    acc_train = accuracy_fn(preds_train, train_labels)
    f1_train  = macrof1_fn(preds_train, train_labels)
    acc_test  = accuracy_fn(preds_test,  test_labels)
    f1_test   = macrof1_fn(preds_test,   test_labels)

    # Report
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

    Arguments:
        method_obj: any object with fit() and predict()
        train_features, train_labels: training data
        test_features, test_labels: test/validation data
    Returns:
        dict with keys: train_mse, test_mse, train_time, pred_time
    """
    # Train
    t0 = time.time()
    preds_train = method_obj.fit(train_features, train_labels)
    train_time  = time.time() - t0

    # Predict
    t0 = time.time()
    preds_test  = method_obj.predict(test_features)
    pred_time   = time.time() - t0

    # Metrics
    train_mse = mse_fn(preds_train, train_labels)
    test_mse  = mse_fn(preds_test,  test_labels)

    # Report
    print(f"\n{'='*52}")
    print(f"  Task : Regression")
    print(f"{'='*52}")
    print(f"  Train MSE : {train_mse:.6f}")
    print(f"  Test  MSE : {test_mse:.6f}")
    print(f"  Train time : {train_time:.4f}s  |  Predict time : {pred_time:.4f}s")

    return dict(train_mse=train_mse, test_mse=test_mse,
                train_time=train_time, pred_time=pred_time)


# =============================================================================
# HELPERS
# =============================================================================

def get_score(preds, labels, task):
    """
    Returns a scalar score where LOWER is always better.
    - Regression    : MSE
    - Classification: error rate (1 - accuracy)
    """
    if task == "regression":
        return mse_fn(preds, labels)
    elif task == "classification":
        return 1.0 - accuracy_fn(preds, labels) / 100.0
    else:
        raise ValueError(f"Unknown task: {task}")


# =============================================================================
# HYPERPARAMETER SEARCH: VALIDATION SET
# =============================================================================

def validation_set_search(method_class, method_kwargs_list,
                           train_features, train_labels,
                           val_features,   val_labels, task):
    """
    Simple validation set hyperparameter search.

    Arguments:
        method_class       : class to instantiate (e.g. LinearRegression)
        method_kwargs_list : list of dicts, one per hyperparameter combo
                             e.g. [{'lambda_reg': 0}, {'lambda_reg': 0.1}]
        task               : 'regression' or 'classification'
    Returns:
        best_kwargs (dict), results (list of dicts with 'kwargs' and 'score')
    """
    print(f"\n--- Validation set search ({len(method_kwargs_list)} configs) ---")
    results = []

    for kwargs in method_kwargs_list:
        model = method_class(**kwargs)
        model.fit(train_features, train_labels)
        score = get_score(model.predict(val_features), val_labels, task)
        results.append({'kwargs': kwargs, 'score': score})
        print(f"  {kwargs}  →  score = {score:.6f}")

    best = min(results, key=lambda x: x['score'])
    print(f"\n  Best : {best['kwargs']}  (score = {best['score']:.6f})")
    return best['kwargs'], results


# =============================================================================
# HYPERPARAMETER SEARCH: K-FOLD CROSS VALIDATION
# =============================================================================

def kfold_cross_validation(method_class, method_kwargs_list,
                            features, labels, task, k=5):
    """
    K-Fold cross-validation hyperparameter search.
    Works with ANY method that has fit() and predict().

    Arguments:
        method_class       : class to instantiate
        method_kwargs_list : list of dicts, one per hyperparameter combo
        features, labels   : full training set (not test!)
        task               : 'regression' or 'classification'
        k                  : number of folds (5 or 10 recommended)
    Returns:
        best_kwargs (dict), results (list of dicts)
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
# LOOCV SHORTCUT (linear regression only)
# =============================================================================

def loocv_linear_regression(features, labels, lambda_reg):
    """
    Exact LOOCV for linear regression using the hat matrix shortcut.
    Cost: one fit + O(N²)  — no need to train N separate models.

    Formula:  LOO_error_i = residual_i / (1 - H_ii)
    where H = X(XᵀX + λI)⁻¹Xᵀ  is the hat matrix.
    """
    N, D = features.shape
    X    = np.hstack([features, np.ones((N, 1))])   # add bias

    I         = np.eye(D + 1)
    I[-1, -1] = 0                                    # don't regularize bias
    XtX_reg   = X.T @ X + lambda_reg * I

    # Hat matrix: avoid explicit inverse — use solve
    H         = X @ np.linalg.solve(XtX_reg, X.T)  # (N, N)

    # Fit once
    w         = np.linalg.solve(XtX_reg, X.T @ labels)
    residuals = labels - X @ w                       # (N,)

    # LOO shortcut
    h_diag    = np.diag(H)                           # (N,)
    loo_err   = residuals / (1 - h_diag)
    return float(np.mean(loo_err ** 2))


# =============================================================================
# COMPARE ALL CV METHODS (for report visualization)
# =============================================================================

def compare_cv_methods(features, labels, lambdas, k=5):
    """
    Runs validation set, K-Fold, and LOOCV for linear regression
    across a range of λ values. Useful for a report comparison plot.

    Returns:
        dict with keys: lambdas, val, kfold_mean, kfold_std, loocv
    """
    from src.methods.linear_regression import LinearRegression

    N        = len(features)
    val_size = int(0.2 * N)
    X_tr, y_tr = features[:-val_size], labels[:-val_size]
    X_val, y_val = features[-val_size:], labels[-val_size:]

    val_scores, kfold_means, kfold_stds, loocv_scores = [], [], [], []

    print(f"\n{'λ':<12} {'Val MSE':<16} {f'{k}-Fold MSE':<26} LOOCV MSE")
    print("-" * 65)

    for lam in lambdas:
        # 1. Validation set
        m = LinearRegression(lambda_reg=lam)
        m.fit(X_tr, y_tr)
        val_s = mse_fn(m.predict(X_val), y_val)
        val_scores.append(val_s)

        # 2. K-Fold
        _, res = kfold_cross_validation(
            LinearRegression, [{'lambda_reg': lam}],
            features, labels, 'regression', k=k
        )
        kfold_means.append(res[0]['mean'])
        kfold_stds.append(res[0]['std'])

        # 3. LOOCV shortcut
        loo = loocv_linear_regression(features, labels, lam)
        loocv_scores.append(loo)

        print(f"λ={lam:<10} {val_s:<16.6f} "
              f"{res[0]['mean']:.6f} ± {res[0]['std']:.4f}     {loo:.6f}")

    return dict(lambdas=lambdas, val=val_scores,
                kfold_mean=kfold_means, kfold_std=kfold_stds,
                loocv=loocv_scores)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_cv_comparison(cv_results, save_path="cv_comparison.png"):
    """
    Plots val set, K-Fold, and LOOCV scores vs lambda on the same axes.
    Pass the dict returned by compare_cv_methods().
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    lambdas = cv_results['lambdas']
    x       = np.arange(len(lambdas))
    labels  = [str(l) for l in lambdas]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, cv_results['val'],        marker='o', label='Validation set')
    ax.errorbar(x, cv_results['kfold_mean'],
                yerr=cv_results['kfold_std'],
                marker='s', capsize=4,   label='K-Fold CV')
    ax.plot(x, cv_results['loocv'],      marker='^', label='LOOCV')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("λ (regularization strength)")
    ax.set_ylabel("MSE")
    ax.set_title("Hyperparameter search: comparing CV methods")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


def plot_loss_curve(loss_history, save_path="loss_curve.png"):
    """Plots gradient descent convergence curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE loss")
    plt.title("Gradient descent convergence")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to {save_path}")