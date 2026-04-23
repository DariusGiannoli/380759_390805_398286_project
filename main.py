import argparse
import os
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn
from src import evaluation
from src import optimization

np.random.seed(100)


# =============================================================================
# EXPERIMENTS MODE — generates all report figures in one run
# =============================================================================

def run_experiments(train_features, train_labels_classif, train_labels_reg):
    """Runs the full set of experiments used for the report."""
    os.makedirs("plots", exist_ok=True)

    # -------------------------------------------------------------------
    # Linear Regression
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LINEAR REGRESSION")
    print("=" * 60)

    # HP sweep: lambda
    sweep = hyperparameter_sweep(
        LinearRegression, 'lambda_reg',
        [0, 0.001, 0.01, 0.1, 1, 10, 100],
        fixed_kwargs={}, features=train_features, labels=train_labels_reg,
        task='regression', k=5,
    )
    plot_hyperparameter_sweep(sweep, 'regression', log_x=False,
                              save_path="plots/linreg_lambda_sweep.png")

    # Ablation: degree + interactions
    ablation = ablation_comparison(
        LinearRegression,
        configs=[
            ("vanilla",        {}),
            ("L2 reg λ=0.1",   {'lambda_reg': 0.1}),
            ("degree=2",       {'degree': 2}),
            ("degree=2 + int.",{'degree': 2, 'interaction': True}),
            ("degree=3",       {'degree': 3}),
        ],
        features=train_features, labels=train_labels_reg,
        task='regression', k=5,
    )
    plot_ablation(ablation, 'regression', title="Linear Regression ablation",
                  save_path="plots/linreg_ablation.png")

    # Learning curve
    lc = compute_learning_curve(
        LinearRegression, {'lambda_reg': 0.1},
        train_features, train_labels_reg, 'regression',
    )
    plot_learning_curve(lc, 'regression', title="Linear Regression — learning curve",
                        save_path="plots/linreg_learning_curve.png")

    # CV method comparison (val vs k-fold vs LOOCV)
    cv_cmp = compare_cv_methods(
        train_features, train_labels_reg,
        lambdas=[0, 0.001, 0.01, 0.1, 1, 10, 100], k=5,
    )
    plot_cv_comparison(cv_cmp, save_path="plots/linreg_cv_comparison.png")

    # -------------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LOGISTIC REGRESSION")
    print("=" * 60)

    sweep = hyperparameter_sweep(
        LogisticRegression, 'lr',
        [1e-4, 1e-3, 1e-2, 1e-1],
        fixed_kwargs={'max_iters': 500},
        features=train_features, labels=train_labels_classif,
        task='classification', k=5,
    )
    plot_hyperparameter_sweep(sweep, 'classification', log_x=True,
                              save_path="plots/logreg_lr_sweep.png")

    ablation = ablation_comparison(
        LogisticRegression,
        configs=[
            ("vanilla GD (zeros)",  {'lr': 1e-2, 'max_iters': 500}),
            ("xavier init",         {'lr': 1e-2, 'max_iters': 500, 'init': 'xavier'}),
            ("random init",         {'lr': 1e-2, 'max_iters': 500, 'init': 'random'}),
            ("+ momentum (β=0.9)",  {'lr': 1e-2, 'max_iters': 500, 'beta': 0.9}),
            ("+ early stopping",    {'lr': 1e-2, 'max_iters': 500, 'tol': 1e-6}),
            ("all enhancements",    {'lr': 1e-2, 'max_iters': 500, 'beta': 0.9,
                                     'tol': 1e-6, 'init': 'xavier'}),
        ],
        features=train_features, labels=train_labels_classif,
        task='classification', k=5,
    )
    plot_ablation(ablation, 'classification', title="Logistic Regression ablation",
                  save_path="plots/logreg_ablation.png")

    lc = compute_learning_curve(
        LogisticRegression, {'lr': 1e-2, 'max_iters': 500},
        train_features, train_labels_classif, 'classification',
    )
    plot_learning_curve(lc, 'classification', title="Logistic Regression — learning curve",
                        save_path="plots/logreg_learning_curve.png")

    # Confusion matrix on held-out validation
    N = len(train_features)
    val_size = int(0.2 * N)
    X_tr, y_tr = train_features[:-val_size], train_labels_classif[:-val_size]
    X_val, y_val = train_features[-val_size:], train_labels_classif[-val_size:]
    m = LogisticRegression(lr=0.3, max_iters=1000, beta=0.9)
    m.fit(X_tr, y_tr)
    plot_confusion_matrix(m.predict(X_val), y_val,
                          class_names=['Low', 'Medium', 'High'],
                          save_path="plots/logreg_confusion_matrix.png")

    # Stratified vs standard K-fold
    print("\n--- Stratified vs standard K-Fold (logistic regression) ---")
    stratified_kfold_cross_validation(
        LogisticRegression, [{'lr': 1e-2, 'max_iters': 500}],
        train_features, train_labels_classif, k=5,
    )

    # -------------------------------------------------------------------
    # KNN (classification + regression)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  KNN")
    print("=" * 60)

    # K sweep — classification
    sweep = hyperparameter_sweep(
        KNN, 'k', [1, 3, 5, 7, 10, 15, 20, 30],
        fixed_kwargs={'task_kind': 'classification'},
        features=train_features, labels=train_labels_classif,
        task='classification', k=5,
    )
    plot_hyperparameter_sweep(sweep, 'classification', log_x=False,
                              save_path="plots/knn_k_classif.png")

    # K sweep — regression
    sweep = hyperparameter_sweep(
        KNN, 'k', [1, 3, 5, 7, 10, 15, 20, 30],
        fixed_kwargs={'task_kind': 'regression'},
        features=train_features, labels=train_labels_reg,
        task='regression', k=5,
    )
    plot_hyperparameter_sweep(sweep, 'regression', log_x=False,
                              save_path="plots/knn_k_reg.png")

    # Ablation: weighted + metric
    ablation = ablation_comparison(
        KNN,
        configs=[
            ("k=5, l2, uniform",   {'k': 5, 'task_kind': 'classification'}),
            ("k=5, l1",            {'k': 5, 'task_kind': 'classification', 'metric': 'l1'}),
            ("k=5, cosine",        {'k': 5, 'task_kind': 'classification', 'metric': 'cosine'}),
            ("k=5, weighted",      {'k': 5, 'task_kind': 'classification', 'weighted': True}),
            ("k=5, weighted+l1",   {'k': 5, 'task_kind': 'classification', 'weighted': True, 'metric': 'l1'}),
        ],
        features=train_features, labels=train_labels_classif,
        task='classification', k=5,
    )
    plot_ablation(ablation, 'classification', title="KNN ablation (classification)",
                  save_path="plots/knn_ablation.png")

    # Timing scaling — KNN predict is O(N)
    timing = timing_comparison(
        KNN, {'k': 5, 'task_kind': 'classification'},
        train_features, train_labels_classif,
    )
    plot_timing(timing, title="KNN timing vs training size",
                save_path="plots/knn_timing.png")

    print("\n" + "=" * 60)
    print("  All experiment plots saved in ./plots/")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    """
    Arguments:
        args (Namespace): parsed command-line arguments.
    """
    # -------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")

    feature_data = np.load(args.data_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, \
    train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],
        feature_data['xtest'],
        feature_data['ytrainreg'],
        feature_data['ytestreg'],
        feature_data['ytrainclassif'],
        feature_data['ytestclassif'],
    )

    print(f"Train features : {train_features.shape}")
    print(f"Test features  : {test_features.shape}")

    # -------------------------------------------------------------------
    # 2. PREPARE DATA (normalization + optional validation split)
    # -------------------------------------------------------------------
    mean = np.mean(train_features, axis=0)
    std  = np.std (train_features, axis=0)
    std[std == 0] = 1

    train_features = normalize_fn(train_features, mean, std)
    test_features  = normalize_fn(test_features,  mean, std)

    if not args.test:
        val_size   = int(0.2 * len(train_features))
        train_size = len(train_features) - val_size

        val_features       = train_features[train_size:]
        val_labels_reg     = train_labels_reg[train_size:]
        val_labels_classif = train_labels_classif[train_size:]

        train_features       = train_features[:train_size]
        train_labels_reg     = train_labels_reg[:train_size]
        train_labels_classif = train_labels_classif[:train_size]

        # redirect test_* to validation
        test_features        = val_features
        test_labels_reg      = val_labels_reg
        test_labels_classif  = val_labels_classif
        print(f"Split : {train_size} train / {val_size} val "
              f"(reporting as 'Test' but this is the validation set; pass --test for real test set)")
    else:
        print("Using full training set -> evaluating on test set")

    # -------------------------------------------------------------------
    # 3. EXPERIMENTS MODE — generates all report figures
    # -------------------------------------------------------------------
    if args.experiments:
        run_experiments(train_features, train_labels_classif, train_labels_reg)
        return

    # -------------------------------------------------------------------
    # 3b. TUNE MODE — coarse-to-fine hyperparameter search
    # -------------------------------------------------------------------
    if args.tune:
        if args.method == "linear_regression":
            best, _ = tune_linear_regression(train_features, train_labels_reg, k=5)
        elif args.method == "logistic_regression":
            best, _ = tune_logistic_regression(
                train_features, train_labels_classif, k=5, stratified=True,
            )
        elif args.method == "knn":
            labels = train_labels_reg if args.task == "regression" \
                     else train_labels_classif
            best, _ = tune_knn(train_features, labels, args.task, k_cv=5)
        else:
            raise ValueError(f"--tune not supported for method: {args.method}")
        print(f"\n>>> Best hyperparameters for {args.method}: {best}")
        return

    # -------------------------------------------------------------------
    # 4. INITIALIZE METHOD
    # -------------------------------------------------------------------
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task,
                         weighted=args.weighted, metric=args.metric)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(
            lr=args.lr, max_iters=args.max_iters,
            beta=args.beta, tol=args.tol, init=args.init,
        )

    elif args.method == "linear_regression":
        method_obj = LinearRegression(
            lambda_reg=args.lambda_reg, method=args.lr_method,
            lr=args.lr, max_iters=args.max_iters,
            degree=args.degree, interaction=args.interaction,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Method : {args.method}")

    # -------------------------------------------------------------------
    # 5. TRAIN AND EVALUATE
    # -------------------------------------------------------------------
    if args.task == "classification":
        assert args.method != "linear_regression", \
            "Use linear_regression only for regression tasks."
        results = train_and_evaluate_classification(
            method_obj,
            train_features, train_labels_classif,
            test_features,  test_labels_classif,
        )
        if args.confusion:
            os.makedirs("plots", exist_ok=True)
            plot_confusion_matrix(
                method_obj.predict(test_features), test_labels_classif,
                class_names=['Low', 'Medium', 'High'],
                save_path="plots/confusion_matrix.png",
            )

    elif args.task == "regression":
        assert args.method != "logistic_regression", \
            "Use logistic_regression only for classification tasks."
        results = train_and_evaluate_regression(
            method_obj,
            train_features, train_labels_reg,
            test_features,  test_labels_reg,
        )

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # -------------------------------------------------------------------
    # 6. OPTIONAL EXTRAS
    # -------------------------------------------------------------------

    # Loss curve for any iterative method
    if len(getattr(method_obj, 'loss_history', [])) > 0:
        os.makedirs("plots", exist_ok=True)
        plot_loss_curve(method_obj.loss_history, save_path="plots/loss_curve.png")

    # Cross-validation hyperparameter search
    if args.cv:
        print(f"\nRunning CV hyperparameter search for {args.method}...")
        lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]

        if args.method == "linear_regression":
            cv_results = compare_cv_methods(
                train_features, train_labels_reg, lambdas, k=5,
            )
            os.makedirs("plots", exist_ok=True)
            plot_cv_comparison(cv_results, save_path="plots/cv_comparison.png")

        elif args.method == "knn":
            k_values    = [1, 3, 5, 7, 10, 15, 20]
            kwargs_list = [{'k': k, 'task_kind': args.task} for k in k_values]
            labels      = train_labels_reg if args.task == "regression" \
                          else train_labels_classif
            best, _     = kfold_cross_validation(
                KNN, kwargs_list, train_features, labels, args.task, k=5,
            )
            print(f"Best K = {best['k']}")

        elif args.method == "logistic_regression":
            lr_values   = [1e-4, 1e-3, 1e-2, 1e-1]
            kwargs_list = [{'lr': lr, 'max_iters': args.max_iters}
                           for lr in lr_values]
            best, _     = kfold_cross_validation(
                LogisticRegression, kwargs_list,
                train_features, train_labels_classif,
                'classification', k=5,
            )
            print(f"Best lr = {best['lr']}")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Task & method
    parser.add_argument("--task",        default="classification", type=str,
                        help="classification / regression")
    parser.add_argument("--method",      default="dummy_classifier", type=str,
                        help="dummy_classifier / knn / logistic_regression / linear_regression")
    parser.add_argument("--data_path",   default="data/features.npz", type=str,
                        help="path to dataset .npz file")

    # KNN
    parser.add_argument("--K",           default=1,    type=int,
                        help="number of neighbors for KNN")
    parser.add_argument("--weighted",    action="store_true",
                        help="use distance-weighted voting for KNN")
    parser.add_argument("--metric",      default="l2", type=str,
                        help="distance metric for KNN: l2 / l1 / cosine")

    # Iterative methods (logistic reg + GD linear reg)
    parser.add_argument("--lr",          default=1e-3, type=float,
                        help="learning rate for iterative methods")
    parser.add_argument("--max_iters",   default=500,  type=int,
                        help="max iterations for iterative methods")

    # Linear regression
    parser.add_argument("--lambda_reg",  default=0.0,  type=float,
                        help="L2 regularization strength")
    parser.add_argument("--lr_method",   default="closed_form", type=str,
                        help="linear regression method: closed_form / gradient_descent")
    parser.add_argument("--degree",      default=1,    type=int,
                        help="polynomial degree for linear regression (1 = standard)")
    parser.add_argument("--interaction", action="store_true",
                        help="include cross-term interactions in polynomial expansion")

    # Logistic regression
    parser.add_argument("--beta",        default=0.0,  type=float,
                        help="momentum coefficient for logistic regression (0 = vanilla GD)")
    parser.add_argument("--tol",         default=None, type=float,
                        help="early stopping tolerance for logistic regression")
    parser.add_argument("--init",        default="zeros", type=str,
                        help="weight init for logistic regression: zeros / random / xavier")

    # Evaluation modifiers
    parser.add_argument("--test",        action="store_true",
                        help="evaluate on test set instead of validation set")
    parser.add_argument("--cv",          action="store_true",
                        help="run cross-validation hyperparameter search")
    parser.add_argument("--confusion",   action="store_true",
                        help="generate a confusion matrix (classification only)")
    parser.add_argument("--experiments", action="store_true",
                        help="run all canonical experiments and save report figures")
    parser.add_argument("--tune",        action="store_true",
                        help="run coarse-to-fine hyperparameter search for the selected method")

    args = parser.parse_args()
    main(args)
