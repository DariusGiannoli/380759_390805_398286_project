import numpy as np
from itertools import combinations_with_replacement

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Multinomial (softmax) logistic regression with cross-entropy loss,
    optimized by gradient descent with optional heavy-ball momentum,
    L2 regularization (bias excluded), polynomial feature expansion,
    class-weighted CE, mini-batch SGD and inverse-time learning-rate
    decay.
    """

    def __init__(self, lr, max_iters=500, beta=0.0, tol=None, init='zeros',
                 degree=1, interaction=False, lambda_reg=0.0, class_weight=None,
                 batch_size=None, lr_decay=0.0):
        """
        Arguments:
            lr (float): base learning rate
            max_iters (int): total number of gradient updates
            beta (float): heavy-ball momentum coefficient
            tol (float | None): early-stopping tolerance on |Δloss|
            init (str): 'zeros' | 'random' | 'xavier'
            degree (int): 1 = linear, 2+ = polynomial feature expansion
            interaction (bool): False = per-feature powers, True = full cross-terms
            lambda_reg (float): L2 regularization strength (bias not regularized)
            class_weight (None | 'balanced' | array-like): per-class CE weights
            batch_size (None | int): None / >=N → full-batch GD;
                int < N → mini-batch SGD (random batch each step, shuffled epochs)
            lr_decay (float): inverse-time decay  η_t = lr / (1 + lr_decay · t).
                              0 = constant learning rate.
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None
        self.beta = beta
        self.tol = tol
        self.init = init
        self.degree = degree
        self.interaction = interaction
        self.lambda_reg = lambda_reg
        self.class_weight = class_weight
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.loss_history = []

    def _expand_features(self, X):
        """
        Polynomial feature expansion (same convention as LinearRegression).
        degree=1 : returns X unchanged
        degree>1, interaction=False : adds x^2, x^3, ... per feature (no cross-terms)
        degree>1, interaction=True  : full expansion including cross-terms
        """
        if self.degree == 1:
            return X
        D = X.shape[1]
        features = [X]
        if self.interaction:
            for d in range(2, self.degree + 1):
                for combo in combinations_with_replacement(range(D), d):
                    features.append(np.prod(X[:, combo], axis=1, keepdims=True))
        else:
            for d in range(2, self.degree + 1):
                features.append(X ** d)
        return np.hstack(features)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): class labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        N = training_data.shape[0]
        C = get_n_classes(training_labels)

        X_poly = self._expand_features(training_data)
        # Add bias column (last column = 1 so its weight becomes the intercept)
        X = np.hstack([X_poly, np.ones((N, 1))])    # (N, D_poly + 1)
        Y = label_to_onehot(training_labels, C)     # (N, C)

        # Per-class weights (for imbalanced cross-entropy)
        if self.class_weight is None:
            cw = np.ones(C)
        elif isinstance(self.class_weight, str) and self.class_weight == 'balanced':
            counts = np.bincount(training_labels.astype(int), minlength=C)
            cw = N / (C * np.maximum(counts, 1))
        else:
            cw = np.asarray(self.class_weight, dtype=float)
        s = (Y * cw[None, :]).sum(axis=1, keepdims=True)   # per-sample weight, (N,1)

        # Weight initialization
        input_dim = X.shape[1]
        if self.init == 'zeros':
            self.weights = np.zeros((input_dim, C))
        elif self.init == 'random':
            self.weights = np.random.randn(input_dim, C)
        elif self.init == 'xavier':
            self.weights = np.random.randn(input_dim, C) * np.sqrt(2.0 / input_dim)
        else:
            raise ValueError(f"init must be 'zeros', 'random' or 'xavier'")

        velocity = np.zeros_like(self.weights)
        self.loss_history = []
        prev_loss = np.inf

        # Mini-batch sampler: shuffled epochs, advance pointer per step
        bs = N if (self.batch_size is None or self.batch_size >= N) else int(self.batch_size)
        perm = np.random.permutation(N)
        ptr  = 0

        for t in range(self.max_iters):
            # Choose batch indices
            if bs < N:
                if ptr + bs > N:                     # next epoch
                    perm = np.random.permutation(N)
                    ptr = 0
                idx = perm[ptr:ptr + bs]
                ptr += bs
            else:
                idx = slice(None)

            X_b, Y_b, s_b = X[idx], Y[idx], s[idx]

            # Softmax on batch
            logits = X_b @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)

            # Loss tracked on the batch (cheap, noisy for SGD as expected)
            ce = - np.sum(s_b * np.sum(Y_b * np.log(probs + 1e-15), axis=1, keepdims=True)) / bs
            l2 = 0.5 * self.lambda_reg * np.sum(self.weights[:-1] ** 2) / N
            loss = ce + l2
            self.loss_history.append(loss)

            #(Optional) Early stopping (only sensible in full-batch mode)
            if self.tol is not None and bs == N and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            # Gradient of weighted CE on batch + L2
            grad = X_b.T @ (s_b * (probs - Y_b)) / bs
            if self.lambda_reg > 0:
                grad[:-1] += (self.lambda_reg / N) * self.weights[:-1]

            # Inverse-time learning-rate schedule
            lr_t = self.lr / (1.0 + self.lr_decay * t)

            # Heavy-ball momentum
            velocity = self.beta * velocity + grad
            self.weights -= lr_t * velocity

        return self.predict(training_data)

    def predict_proba(self, test_data):
        """
        Returns the (N, C) softmax probability matrix for the test data.
        Used for ROC/AUC analysis.
        """
        N = test_data.shape[0]
        X_poly = self._expand_features(test_data)
        X = np.hstack([X_poly, np.ones((N, 1))])
        logits = X @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        return onehot_to_label(self.predict_proba(test_data))