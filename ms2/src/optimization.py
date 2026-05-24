"""Hyperparameter search utilities: single-parameter sweeps and grid search."""

import numpy as np
from itertools import product

from src.evaluation import (
    kfold_cross_validation,
    stratified_kfold_cross_validation,
)


def hyperparameter_sweep(method_class, param_name, param_values, fixed_kwargs,
                         features, labels, task, k=5, stratified=False,
                         verbose=True):
    """
    Sweep a single hyperparameter and run k-fold CV for each value.

    Returns a dict with:
        param_name, param_values,
        means / stds  (loss-like score, lower = better),
        f1_means / f1_stds  (classification only).
    """
    kwargs_list = [dict(fixed_kwargs, **{param_name: v}) for v in param_values]
    if stratified and task == 'classification':
        _, results = stratified_kfold_cross_validation(
            method_class, kwargs_list, features, labels, k=k, verbose=verbose
        )
    else:
        _, results = kfold_cross_validation(
            method_class, kwargs_list, features, labels, task, k=k,
            verbose=verbose,
        )
    out = dict(
        param_name=param_name,
        param_values=list(param_values),
        means=np.array([r['mean'] for r in results]),
        stds =np.array([r['std']  for r in results]),
    )
    if task == 'classification':
        out['f1_means'] = np.array([r['f1_mean'] for r in results])
        out['f1_stds']  = np.array([r['f1_std']  for r in results])
    return out


def grid_search(method_class, param_grid, features, labels, task,
                k=5, fixed_kwargs=None, stratified=False, verbose=True):
    """Full Cartesian-product grid search via k-fold CV."""
    fixed_kwargs = fixed_kwargs or {}
    names = list(param_grid.keys())
    value_lists = [param_grid[p] for p in names]
    combos = list(product(*value_lists))
    if verbose:
        print(f"  -- grid: {len(combos)} combos over {names}")
    kwargs_list = [dict(fixed_kwargs, **dict(zip(names, combo))) for combo in combos]
    if stratified and task == 'classification':
        _, results = stratified_kfold_cross_validation(
            method_class, kwargs_list, features, labels, k=k, verbose=verbose
        )
    else:
        _, results = kfold_cross_validation(
            method_class, kwargs_list, features, labels, task, k=k,
            verbose=verbose,
        )
    sorted_r = sorted(results, key=lambda r: r['mean'])
    return sorted_r, sorted_r[0]['kwargs']
