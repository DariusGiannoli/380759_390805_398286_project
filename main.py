import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        pass

    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        ### WRITE YOUR CODE HERE
        pass

    elif args.method == "logistic_regression":
        ### WRITE YOUR CODE HERE
        pass

    elif args.method == "linear_regression":
        ### WRITE YOUR CODE HERE
        pass

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_classif)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.task == "regression":
        assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_reg)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
