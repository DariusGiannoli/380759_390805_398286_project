"""
Hyperparameter optimization for ML methods.

This module is responsible for *searching* for the best hyperparameters.
For pure model evaluation / scoring / CV infrastructure, see evaluation.py.
"""

import numpy as np
from itertools import product

from src.evaluation import (
    get_score, kfold_cross_validation, stratified_kfold_cross_validation,
    loocv_linear_regression,
)
from src.utils import mse_fn


# =============================================================================
# VALIDATION SET SEARCH
# =============================================================================

def validation_set_search(method_class, method_kwargs_list,
                           train_features, train_labels,
                           val_features, val_labels, task):
    """
    Simple validation set hyperparameter search.

    Arguments:
        method_class       : class to instantiate (e.g. LinearRegression)
        method_kwargs_list : list of dicts, one per hyperparameter combo
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
# 1-D HYPERPARAMETER SWEEP
# =============================================================================

def hyperparameter_sweep(method_class, param_name, param_values, fixed_kwargs,
                         features, labels, task, k=5, stratified=False):
    """
    Sweeps a single hyperparameter, runs k-fold CV for each value.

    Returns:
        dict with keys: param_name, param_values, means, stds
    """
    kwargs_list = [dict(fixed_kwargs, **{param_name: v}) for v in param_values]

    if stratified and task == 'classification':
        _, results = stratified_kfold_cross_validation(
            method_class, kwargs_list, features, labels, k=k
        )
    else:
        _, results = kfold_cross_validation(
            method_class, kwargs_list, features, labels, task, k=k
        )

    return dict(
        param_name=param_name,
        param_values=list(param_values),
        means=np.array([r['mean'] for r in results]),
        stds =np.array([r['std']  for r in results]),
    )


def plot_hyperparameter_sweep(sweep_results, task, log_x=False,
                               save_path="plots/hp_sweep.png"):
    """Plots 1-D HP sweep results with error bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    values = sweep_results['param_values']
    means  = sweep_results['means']
    stds   = sweep_results['stds']
    name   = sweep_results['param_name']
    ylab   = "MSE" if task == "regression" else "Error rate (1 - acc)"

    fig, ax = plt.subplots(figsize=(7, 4))
    numeric = all(isinstance(v, (int, float)) for v in values)
    x = values if numeric else np.arange(len(values))

    ax.errorbar(x, means, yerr=stds, marker='o', capsize=4)

    if log_x and numeric and all(v > 0 for v in values):
        ax.set_xscale('log')
    if not numeric:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values])

    best_idx = int(np.argmin(means))
    ax.scatter([x[best_idx]], [means[best_idx]], color='red', s=80, zorder=3,
               label=f"best: {name}={values[best_idx]}")

    ax.set_xlabel(name)
    ax.set_ylabel(ylab)
    ax.set_title(f"Hyperparameter sweep: {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"HP sweep plot saved to {save_path}")


# =============================================================================
# MULTI-DIMENSIONAL GRID SEARCH
# =============================================================================

def grid_search(method_class, param_grid, features, labels, task,
                k=5, fixed_kwargs=None, stratified=False, verbose=True):
    """
    Full Cartesian-product grid search via k-fold CV.

    Arguments:
        param_grid   : dict {param_name: [values]}
        fixed_kwargs : params that stay constant across the grid
        stratified   : use stratified k-fold (classification only)

    Returns:
        sorted_results (list of dicts, best first),
        best_kwargs (dict)
    """
    fixed_kwargs = fixed_kwargs or {}
    param_names = list(param_grid.keys())
    value_lists = [param_grid[p] for p in param_names]

    combos = list(product(*value_lists))
    if verbose:
        print(f"\n--- Grid search: {len(combos)} combinations over {param_names} ---")

    kwargs_list = [
        dict(fixed_kwargs, **dict(zip(param_names, combo)))
        for combo in combos
    ]

    cv_fn = stratified_kfold_cross_validation if (stratified and task == 'classification') \
            else kfold_cross_validation
    if stratified and task == 'classification':
        _, results = cv_fn(method_class, kwargs_list, features, labels, k=k)
    else:
        _, results = cv_fn(method_class, kwargs_list, features, labels, task, k=k)

    sorted_results = sorted(results, key=lambda r: r['mean'])
    best_kwargs = sorted_results[0]['kwargs']

    if verbose:
        print(f"\nTop 5 configurations:")
        for r in sorted_results[:5]:
            print(f"  {r['kwargs']}  →  {r['mean']:.6f} ± {r['std']:.6f}")

    return sorted_results, best_kwargs


# =============================================================================
# COARSE-TO-FINE SEARCH
# =============================================================================

def coarse_to_fine_search(method_class, coarse_grid, fine_spec,
                           features, labels, task,
                           k=5, fixed_kwargs=None, stratified=False):
    """
    Two-stage hyperparameter search:
        1. Coarse grid search over broad ranges
        2. Fine-grained search centered on the winner

    Arguments:
        coarse_grid : dict {param_name: [values]} — broad initial search
        fine_spec   : dict {param_name: callable(best_value) -> list}
                      A function that returns the fine-grained values to try
                      given the winning coarse value for that param.
                      For params not in fine_spec, we keep the coarse winner.
    Returns:
        best_kwargs (dict), all_results (dict with 'coarse' and 'fine')
    """
    print(f"\n{'='*60}")
    print(f"  COARSE-TO-FINE SEARCH: {method_class.__name__}")
    print(f"{'='*60}")

    # Stage 1: coarse
    print("\n>> Stage 1: coarse grid")
    coarse_results, coarse_best = grid_search(
        method_class, coarse_grid, features, labels, task,
        k=k, fixed_kwargs=fixed_kwargs, stratified=stratified,
    )

    # Stage 2: fine
    print(f"\n>> Stage 2: fine grid around coarse winner {coarse_best}")
    fine_grid = {}
    for name, best_val in coarse_best.items():
        if name in (fixed_kwargs or {}):
            continue
        if name in fine_spec:
            fine_grid[name] = fine_spec[name](best_val)
        else:
            fine_grid[name] = [best_val]  # pin to winner

    fine_results, fine_best = grid_search(
        method_class, fine_grid, features, labels, task,
        k=k, fixed_kwargs=fixed_kwargs, stratified=stratified,
    )

    # Pick overall best
    overall_best = fine_results[0] if fine_results[0]['mean'] < coarse_results[0]['mean'] \
                   else coarse_results[0]
    print(f"\n>> Overall best: {overall_best['kwargs']}")
    print(f"   score = {overall_best['mean']:.6f} ± {overall_best['std']:.6f}")

    return overall_best['kwargs'], {'coarse': coarse_results, 'fine': fine_results}


# =============================================================================
# METHOD-SPECIFIC TUNERS (one-call-to-get-best-HP pipelines)
# =============================================================================

def tune_linear_regression(features, labels, k=5):
    """
    Coarse-to-fine tuning for linear regression.
    Searches lambda_reg × degree × interaction.
    """
    from src.methods.linear_regression import LinearRegression

    coarse_grid = {
        'lambda_reg' : [0, 0.001, 0.01, 0.1, 1, 10, 100],
        'degree'     : [1, 2, 3],
        'interaction': [False, True],
    }

    def fine_lambda(best_lam):
        if best_lam == 0:
            return [0, 1e-5, 1e-4, 1e-3]
        return [best_lam * 0.3, best_lam, best_lam * 3]

    fine_spec = {
        'lambda_reg': fine_lambda,
        # degree and interaction stay pinned to coarse winner
    }

    return coarse_to_fine_search(
        LinearRegression, coarse_grid, fine_spec,
        features, labels, 'regression', k=k,
    )


def tune_logistic_regression(features, labels, k=5, stratified=True):
    """
    Coarse-to-fine tuning for logistic regression.
    Searches lr × max_iters × beta × init.
    """
    from src.methods.logistic_regression import LogisticRegression

    coarse_grid = {
        'lr'       : [1e-4, 1e-3, 1e-2, 1e-1],
        'max_iters': [500, 1000],
        'beta'     : [0.0, 0.9],
        'init'     : ['zeros', 'xavier'],
    }

    def fine_lr(best_lr):
        return [best_lr / 3, best_lr, best_lr * 3]

    def fine_iters(best_iters):
        return [best_iters, best_iters * 2]

    fine_spec = {
        'lr'       : fine_lr,
        'max_iters': fine_iters,
    }

    return coarse_to_fine_search(
        LogisticRegression, coarse_grid, fine_spec,
        features, labels, 'classification',
        k=k, stratified=stratified,
    )


def tune_knn(features, labels, task, k_cv=5, stratified=True):
    """
    Coarse-to-fine tuning for KNN.
    Searches k × metric × weighted.
    """
    from src.methods.knn import KNN

    coarse_grid = {
        'k'       : [1, 3, 5, 7, 10, 15, 20, 30, 50],
        'metric'  : ['l1', 'l2', 'cosine'],
        'weighted': [False, True],
    }

    def fine_k(best_k):
        return sorted(set([max(1, best_k - 2), max(1, best_k - 1),
                           best_k, best_k + 1, best_k + 2]))

    fine_spec = {'k': fine_k}

    return coarse_to_fine_search(
        KNN, coarse_grid, fine_spec,
        features, labels, task,
        k=k_cv, fixed_kwargs={'task_kind': task},
        stratified=(stratified and task == 'classification'),
    )


# =============================================================================
# ABLATION COMPARISON
# =============================================================================

def ablation_comparison(method_class, configs, features, labels, task, k=5):
    """
    Compares several named configurations via k-fold CV.

    Arguments:
        configs : list of (name, kwargs) tuples

    Returns:
        list of dicts: {'name', 'kwargs', 'mean', 'std'}
    """
    print(f"\n{'='*60}")
    print(f"  Ablation: {method_class.__name__}")
    print(f"{'='*60}")

    results = []
    for name, kwargs in configs:
        _, r = kfold_cross_validation(
            method_class, [kwargs], features, labels, task, k=k
        )
        results.append({'name': name, 'kwargs': kwargs,
                        'mean': r[0]['mean'], 'std': r[0]['std']})
        print(f"  {name:<30} → {r[0]['mean']:.6f} ± {r[0]['std']:.6f}")

    return results


def plot_ablation(ablation_results, task, title="Ablation",
                  save_path="plots/ablation.png"):
    """Bar plot of ablation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    names = [r['name'] for r in ablation_results]
    means = [r['mean'] for r in ablation_results]
    stds  = [r['std']  for r in ablation_results]
    ylab  = "MSE" if task == "regression" else "Error rate (1 - acc)"

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.0), 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Ablation plot saved to {save_path}")


# =============================================================================
# 2-D GRID HEATMAP
# =============================================================================

def plot_grid_heatmap(grid_results, param_x, param_y,
                      task, save_path="plots/grid_heatmap.png"):
    """
    Plots a 2D heatmap of grid search results. Collapses other params by
    taking the best (lowest) score for each (param_x, param_y) pair.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    x_values = sorted({r['kwargs'][param_x] for r in grid_results},
                      key=lambda v: (isinstance(v, str), v))
    y_values = sorted({r['kwargs'][param_y] for r in grid_results},
                      key=lambda v: (isinstance(v, str), v))

    grid = np.full((len(y_values), len(x_values)), np.inf)
    for r in grid_results:
        xi = x_values.index(r['kwargs'][param_x])
        yi = y_values.index(r['kwargs'][param_y])
        grid[yi, xi] = min(grid[yi, xi], r['mean'])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax,
                 label="MSE" if task == "regression" else "Error rate")

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values], rotation=30, ha='right')
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(v) for v in y_values])
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Grid search: {param_y} vs {param_x}  (best over others)")

    for i in range(len(y_values)):
        for j in range(len(x_values)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}",
                        ha='center', va='center',
                        color='white' if grid[i, j] > grid[np.isfinite(grid)].mean() else 'black',
                        fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Grid heatmap saved to {save_path}")


# =============================================================================
# CV METHOD COMPARISON (val set vs k-fold vs LOOCV)
# =============================================================================

def compare_cv_methods(features, labels, lambdas, k=5):
    """
    Runs validation set, K-Fold, and LOOCV for linear regression
    across a range of lambdas. Useful for report comparison.

    Returns:
        dict with keys: lambdas, val, kfold_mean, kfold_std, loocv
    """
    from src.methods.linear_regression import LinearRegression

    N        = len(features)
    val_size = int(0.2 * N)
    X_tr, y_tr   = features[:-val_size], labels[:-val_size]
    X_val, y_val = features[-val_size:], labels[-val_size:]

    val_scores, kfold_means, kfold_stds, loocv_scores = [], [], [], []

    print(f"\n{'λ':<12} {'Val MSE':<16} {f'{k}-Fold MSE':<26} LOOCV MSE")
    print("-" * 65)

    for lam in lambdas:
        m = LinearRegression(lambda_reg=lam)
        m.fit(X_tr, y_tr)
        val_s = mse_fn(m.predict(X_val), y_val)
        val_scores.append(val_s)

        _, res = kfold_cross_validation(
            LinearRegression, [{'lambda_reg': lam}],
            features, labels, 'regression', k=k,
        )
        kfold_means.append(res[0]['mean'])
        kfold_stds.append(res[0]['std'])

        loo = loocv_linear_regression(features, labels, lam)
        loocv_scores.append(loo)

        print(f"λ={lam:<10} {val_s:<16.6f} "
              f"{res[0]['mean']:.6f} ± {res[0]['std']:.4f}     {loo:.6f}")

    return dict(lambdas=lambdas, val=val_scores,
                kfold_mean=kfold_means, kfold_std=kfold_stds,
                loocv=loocv_scores)


def plot_cv_comparison(cv_results, save_path="plots/cv_comparison.png"):
    """Plots val set, K-Fold, and LOOCV scores vs lambda."""
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
    plt.close()
    print(f"Plot saved to {save_path}")
