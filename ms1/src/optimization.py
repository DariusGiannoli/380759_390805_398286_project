"""Hyperparameter search utilities: sweeps, grids, coarse-to-fine, CV-method comparison."""

import numpy as np
from itertools import product

from src.evaluation import (
    kfold_cross_validation, stratified_kfold_cross_validation,
    loocv_linear_regression,
)
from src.utils import mse_fn


def hyperparameter_sweep(method_class, param_name, param_values, fixed_kwargs,
                         features, labels, task, k=5, stratified=False):
    """
    Sweep a single hyperparameter and run k-fold CV for each value.
    Returns a dict with keys: param_name, param_values, means, stds.
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


def grid_search(method_class, param_grid, features, labels, task,
                k=5, fixed_kwargs=None, stratified=False, verbose=True):
    """Full Cartesian-product grid search via k-fold CV."""
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

    if stratified and task == 'classification':
        _, results = stratified_kfold_cross_validation(
            method_class, kwargs_list, features, labels, k=k
        )
    else:
        _, results = kfold_cross_validation(
            method_class, kwargs_list, features, labels, task, k=k
        )

    sorted_results = sorted(results, key=lambda r: r['mean'])
    best_kwargs = sorted_results[0]['kwargs']

    if verbose:
        print(f"\nTop 5 configurations:")
        for r in sorted_results[:5]:
            print(f"  {r['kwargs']}  →  {r['mean']:.6f} ± {r['std']:.6f}")

    return sorted_results, best_kwargs


def coarse_to_fine_search(method_class, coarse_grid, fine_spec,
                           features, labels, task,
                           k=5, fixed_kwargs=None, stratified=False):
    """
    Two-stage hyperparameter search:
        1. Coarse grid search over broad ranges.
        2. Fine-grained search centered on the winner.

    Arguments:
        coarse_grid : dict {param_name: [values]}
        fine_spec   : dict {param_name: callable(best_value) -> list}
                      A function that returns the fine-grained values to try
                      given the winning coarse value for that param. Params
                      not in fine_spec stay pinned to the coarse winner.
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


def compare_cv_methods(features, labels, lambdas, k=5):
    """
    Run validation-set, k-fold, and LOOCV for linear regression across a
    range of lambdas. Used to check that the three estimators agree.
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
