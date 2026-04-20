import argparse
import numpy as np
import os

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn
from src.evaluation import (
    train_and_evaluate_classification,
    train_and_evaluate_regression,
    kfold_cross_validation,
    compare_cv_methods,
    plot_cv_comparison,
    plot_loss_curve
)

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
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
        feature_data['ytestclassif']
    )

    print(f"Train features : {train_features.shape}")
    print(f"Test features  : {test_features.shape}")

    # =========================================================================
    # 2. PREPARE DATA
    # =========================================================================

    # Normalize using training statistics only (avoid data leakage)
    mean = np.mean(train_features, axis=0)
    std  = np.std(train_features, axis=0)
    std[std == 0] = 1   # avoid division by zero for constant features

    train_features = normalize_fn(train_features, mean, std)
    test_features  = normalize_fn(test_features,  mean, std)

    # Validation split (unless --test flag is passed)
    if not args.test:
        val_size   = int(0.2 * len(train_features))
        train_size = len(train_features) - val_size

        val_features         = train_features[train_size:]
        val_labels_reg       = train_labels_reg[train_size:]
        val_labels_classif   = train_labels_classif[train_size:]

        train_features       = train_features[:train_size]
        train_labels_reg     = train_labels_reg[:train_size]
        train_labels_classif = train_labels_classif[:train_size]

        # Point test variables to validation set
        test_features        = val_features
        test_labels_reg      = val_labels_reg
        test_labels_classif  = val_labels_classif

        print(f"Split : {train_size} train / {val_size} val")
    else:
        print("Using full training set -> evaluating on test set")

    # =========================================================================
    # 3. INITIALIZE METHOD
    # =========================================================================

    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task,
                         weighted=args.weighted, metric=args.metric)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters,
                                        beta=args.beta, tol=args.tol,
                                        init=args.init)

    elif args.method == "linear_regression":
        method_obj = LinearRegression(
            lambda_reg=args.lambda_reg,
            method=args.lr_method,
            lr=args.lr,
            max_iters=args.max_iters,
            degree=args.degree,
            interaction=args.interaction
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Method : {args.method}")

    # =========================================================================
    # 4. TRAIN AND EVALUATE  (logic lives in src/evaluation.py)
    # =========================================================================

    if args.task == "classification":
        assert args.method != "linear_regression", \
            "Use linear_regression only for regression tasks."
        results = train_and_evaluate_classification(
            method_obj,
            train_features, train_labels_classif,
            test_features,  test_labels_classif
        )

    elif args.task == "regression":
        assert args.method != "logistic_regression", \
            "Use logistic_regression only for classification tasks."
        results = train_and_evaluate_regression(
            method_obj,
            train_features, train_labels_reg,
            test_features,  test_labels_reg
        )

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # =========================================================================
    # 5. OPTIONAL EXTRAS
    # =========================================================================

    # Plot gradient descent loss curve
    if len(getattr(method_obj, 'loss_history', [])) > 0:
        plot_loss_curve(method_obj.loss_history)

    # Cross-validation hyperparameter search + comparison plot
    if args.cv:
        print(f"\nRunning CV hyperparameter search for {args.method}...")
        lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]

        if args.method == "linear_regression":
            cv_results = compare_cv_methods(
                train_features, train_labels_reg, lambdas, k=5
            )
            plot_cv_comparison(cv_results)

        elif args.method == "knn":
            k_values    = [1, 3, 5, 7, 10, 15, 20]
            kwargs_list = [{'k': k, 'task_kind': args.task} for k in k_values]
            labels      = train_labels_reg if args.task == "regression" \
                          else train_labels_classif
            best, _     = kfold_cross_validation(
                KNN, kwargs_list, train_features, labels, args.task, k=5
            )
            print(f"Best K = {best['k']}")

        elif args.method == "logistic_regression":
            lr_values   = [1e-4, 1e-3, 1e-2, 1e-1]
            kwargs_list = [{'lr': lr, 'max_iters': args.max_iters}
                           for lr in lr_values]
            best, _     = kfold_cross_validation(
                LogisticRegression, kwargs_list,
                train_features, train_labels_classif,
                'classification', k=5
            )
            print(f"Best lr = {best['lr']}")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task",       default="classification", type=str,
                        help="classification / regression")
    parser.add_argument("--method",     default="dummy_classifier", type=str,
                        help="dummy_classifier / knn / logistic_regression / linear_regression")
    parser.add_argument("--data_path",  default="data/features.npz", type=str,
                        help="path to dataset .npz file")
    parser.add_argument("--K",          default=1,    type=int,
                        help="number of neighbors for KNN")
    parser.add_argument("--weighted",   action="store_true",
                        help="use distance-weighted voting for KNN")
    parser.add_argument("--metric",     default="l2", type=str,
                        help="distance metric for KNN: l2 / l1 / cosine")
    parser.add_argument("--lr",         default=1e-5, type=float,
                        help="learning rate for iterative methods")
    parser.add_argument("--max_iters",  default=100,  type=int,
                        help="max iterations for iterative methods")
    parser.add_argument("--lambda_reg", default=0.0,  type=float,
                        help="L2 regularization strength")
    parser.add_argument("--lr_method",  default="closed_form", type=str,
                        help="linear regression method: closed_form / gradient_descent")
    parser.add_argument("--degree",      default=1,    type=int,
                        help="polynomial degree for linear regression (1 = standard)")
    parser.add_argument("--interaction", action="store_true",
                        help="include cross-term interactions in polynomial expansion")
    parser.add_argument("--beta",        default=0.0,  type=float,
                        help="momentum coefficient for logistic regression (0 = vanilla GD)")
    parser.add_argument("--tol",         default=None, type=float,
                        help="early stopping tolerance for logistic regression")
    parser.add_argument("--init",        default="zeros", type=str,
                        help="weight init for logistic regression: zeros / random / xavier")
    parser.add_argument("--test",       action="store_true",
                        help="evaluate on test set instead of validation set")
    parser.add_argument("--cv",         action="store_true",
                        help="run cross-validation hyperparameter search")

    args = parser.parse_args()
    main(args)