import argparse
import os
import time
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.methods.kmeans import KMeans
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, Identity, Softmax
from src.utils import (
    normalize_fn,
    label_to_onehot,
    onehot_to_label,
    accuracy_fn,
    macrof1_fn,
    mse_fn,
    get_n_classes,
)

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
        feature_data['xtrain'], feature_data['xtest'], feature_data['ytrainreg'],
        feature_data['ytestreg'], feature_data['ytrainclassif'], feature_data['ytestclassif']
    )

    print(f"Train features : {train_features.shape}")
    print(f"Test  features : {test_features.shape}")

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Shuffle BEFORE splitting to avoid leakage if the dataset is ordered
        # (MS1 feedback). RandomState is local so the global seed is preserved.
        rs = np.random.RandomState(0)
        perm = rs.permutation(len(train_features))
        train_features       = train_features[perm]
        train_labels_reg     = train_labels_reg[perm]
        train_labels_classif = train_labels_classif[perm]

        val_size   = int(0.2 * len(train_features))
        train_size = len(train_features) - val_size

        val_features       = train_features[train_size:]
        val_labels_reg     = train_labels_reg[train_size:]
        val_labels_classif = train_labels_classif[train_size:]

        train_features       = train_features[:train_size]
        train_labels_reg     = train_labels_reg[:train_size]
        train_labels_classif = train_labels_classif[:train_size]

        # redirect test_* to validation
        test_features       = val_features
        test_labels_reg     = val_labels_reg
        test_labels_classif = val_labels_classif
        print(f"Split          : {train_size} train / {val_size} val "
              f"(reporting as 'Test' but this is the validation set; pass --test for real test set)")
    else:
        print("Using full training set -> evaluating on test set")

    # z-score normalization using training-set statistics
    mean = np.mean(train_features, axis=0, keepdims=True)
    std  = np.std (train_features, axis=0, keepdims=True)
    std[std == 0] = 1.0
    train_features = normalize_fn(train_features, mean, std)
    test_features  = normalize_fn(test_features,  mean, std)

    ## 3. Initialize the method you want to use.

    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        D = train_features.shape[1]
        hidden_dims = [int(h) for h in args.hidden_dims.split(",") if h.strip()]
        act_name = args.activation.lower()
        hidden_act = ReLU if act_name == "relu" else Sigmoid

        if args.task == "classification":
            C = get_n_classes(train_labels_classif)
            dimensions  = [D] + hidden_dims + [C]
            activations = [hidden_act] * len(hidden_dims) + [Softmax]
        else:  # regression
            dimensions  = [D] + hidden_dims + [1]
            activations = [hidden_act] * len(hidden_dims) + [Identity]

        method_obj = MLP(dimensions=tuple(dimensions), activations=tuple(activations))
        # Stash MLP-specific knobs so the train block below can read them.
        method_obj._epochs        = args.epochs
        method_obj._batch_size    = args.batch_size
        method_obj._learning_rate = args.lr
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Method         : {args.method}")
    print(f"Task           : {args.task}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        train_y = train_labels_classif
        test_y  = test_labels_classif

        t0 = time.time()
        if args.method == "mlp":
            C = get_n_classes(train_y)
            y_one_hot = label_to_onehot(train_y, C)
            loss = MSE if args.loss == "mse" else CrossEntropy
            method_obj.fit(
                train_features, y_one_hot, loss=loss,
                epochs=method_obj._epochs,
                batch_size=method_obj._batch_size,
                learning_rate=method_obj._learning_rate,
            )
            preds_train = onehot_to_label(method_obj.predict(train_features))
        else:
            preds_train = method_obj.fit(train_features, train_y)
        train_time = time.time() - t0

        t0 = time.time()
        if args.method == "mlp":
            preds_test = onehot_to_label(method_obj.predict(test_features))
        else:
            preds_test = method_obj.predict(test_features)
        pred_time = time.time() - t0

        acc_train = accuracy_fn(preds_train, train_y)
        f1_train  = macrof1_fn(preds_train, train_y)
        acc_test  = accuracy_fn(preds_test,  test_y)
        f1_test   = macrof1_fn(preds_test,   test_y)

        print(f"\n{'='*52}")
        print(f"  Task : Classification")
        print(f"{'='*52}")
        print(f"  Train : accuracy = {acc_train:.3f}%  |  F1 = {f1_train:.6f}")
        print(f"  Test  : accuracy = {acc_test:.3f}%  |  F1 = {f1_test:.6f}")
        print(f"  Train time : {train_time:.4f}s  |  Predict time : {pred_time:.4f}s")

    elif args.task == "regression":
        assert args.method != "kmeans", "You should use kmeans as a classification method"

        train_y = train_labels_reg.astype(np.float64)
        test_y  = test_labels_reg.astype(np.float64)

        t0 = time.time()
        if args.method == "mlp":
            y_train_col = train_y.reshape(-1, 1)
            method_obj.fit(
                train_features, y_train_col, loss=MSE,
                epochs=method_obj._epochs,
                batch_size=method_obj._batch_size,
                learning_rate=method_obj._learning_rate,
            )
            preds_train = method_obj.predict(train_features).ravel()
        else:
            preds_train = method_obj.fit(train_features, train_y)
        train_time = time.time() - t0

        t0 = time.time()
        if args.method == "mlp":
            preds_test = method_obj.predict(test_features).ravel()
        else:
            preds_test = method_obj.predict(test_features)
        pred_time = time.time() - t0

        train_mse = mse_fn(preds_train, train_y)
        test_mse  = mse_fn(preds_test,  test_y)
        print(f"\n{'='*52}")
        print(f"  Task : Regression")
        print(f"{'='*52}")
        print(f"  Train MSE : {train_mse:.6f}")
        print(f"  Test  MSE : {test_mse:.6f}")
        print(f"  Train time : {train_time:.4f}s  |  Predict time : {pred_time:.4f}s")

    else:
        raise ValueError(f"--task must be 'classification' or 'regression', got {args.task!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
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
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
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

    # MLP-specific arguments
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="64,32",
        help="comma-separated hidden layer sizes for MLP (e.g. '64,32')",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="hidden-layer activation for MLP: relu / sigmoid",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of training epochs for MLP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="mini-batch size for MLP",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        help="loss for MLP classification: ce (cross-entropy) / mse",
    )

    args = parser.parse_args()
    main(args)
