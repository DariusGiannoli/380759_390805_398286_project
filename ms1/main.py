import argparse
import os
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn
from src.evaluation import (
    train_and_evaluate_classification,
    train_and_evaluate_regression,
)

np.random.seed(100)


def main(args):
    # --- Load data ---
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

    # --- Normalize + (optional) validation split ---
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

    # --- Instantiate the chosen method ---
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
        raise ValueError(f"--method {args.method!r} is not one of: "
                         "dummy_classifier, knn, logistic_regression, linear_regression")

    print(f"Method : {args.method}")

    # --- Train and evaluate ---
    if args.task == "classification":
        assert args.method != "linear_regression", \
            "Use linear_regression only for regression tasks."
        train_and_evaluate_classification(
            method_obj,
            train_features, train_labels_classif,
            test_features,  test_labels_classif,
        )

    elif args.task == "regression":
        assert args.method != "logistic_regression", \
            "Use logistic_regression only for classification tasks."
        train_and_evaluate_regression(
            method_obj,
            train_features, train_labels_reg,
            test_features,  test_labels_reg,
        )

    else:
        raise ValueError(f"--task must be 'classification' or 'regression', got {args.task!r}")


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

    # Evaluation modifier
    parser.add_argument("--test",        action="store_true",
                        help="evaluate on test set instead of validation set")

    args = parser.parse_args()
    main(args)
