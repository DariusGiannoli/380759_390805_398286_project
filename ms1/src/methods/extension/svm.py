"""
Linear multi-class SVM trained with sub-gradient descent on the
Crammer-Singer hinge loss.

For each sample (x_i, y_i):

    L_i(W) = max( 0,  1 + max_{c ≠ y_i}(w_c^T x_i)  -  w_{y_i}^T x_i )

plus an L2 regularizer (1/2) λ ‖W‖² on non-bias rows.

This shares the linear-classifier family with Logistic Regression but
optimizes a *margin* objective (hinge) rather than a *probabilistic* one
(log-loss). The two models have identical capacity, so a comparison
isolates the effect of the loss function on this dataset, and provides a
hyperplane-margin perspective alongside the maximum-likelihood view of
LogReg.
"""

import numpy as np

from ...utils import get_n_classes


class LinearSVM(object):
    """
    Linear multi-class SVM (Crammer-Singer hinge loss) trained by
    sub-gradient descent, with optional heavy-ball momentum, L2
    regularization (bias excluded), inverse-frequency class weighting,
    mini-batch SGD and inverse-time learning-rate decay.
    """

    def __init__(self, lr=0.01, max_iters=1000, lambda_reg=1e-3,
                 batch_size=None, lr_decay=0.0, beta=0.0,
                 class_weight=None):
        """
        Arguments mirror LogisticRegression so the two are HP-comparable.
            lr (float)         : base learning rate
            max_iters (int)    : total sub-gradient updates
            lambda_reg (float) : L2 strength on non-bias rows
            batch_size (None|int): full-batch (None) or mini-batch SGD
            lr_decay (float)   : η_t = lr / (1 + lr_decay · t)
            beta (float)       : heavy-ball momentum
            class_weight (None|'balanced'|array): per-class loss multiplier
        """
        self.lr = lr
        self.max_iters = max_iters
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.beta = beta
        self.class_weight = class_weight
        self.weights = None
        self.loss_history = []

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): class labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        N, D = training_data.shape
        C = get_n_classes(training_labels)
        y = training_labels.astype(int)

        X = np.hstack([training_data, np.ones((N, 1))])      # (N, D+1)

        if self.class_weight is None:
            cw = np.ones(C)
        elif isinstance(self.class_weight, str) and self.class_weight == 'balanced':
            counts = np.bincount(y, minlength=C)
            cw = N / (C * np.maximum(counts, 1))
        else:
            cw = np.asarray(self.class_weight, dtype=float)

        self.weights = np.zeros((D + 1, C))
        velocity = np.zeros_like(self.weights)
        self.loss_history = []

        bs = N if (self.batch_size is None or self.batch_size >= N) else int(self.batch_size)
        perm = np.random.permutation(N)
        ptr = 0

        for t in range(self.max_iters):
            if bs < N:
                if ptr + bs > N:
                    perm = np.random.permutation(N)
                    ptr = 0
                idx = perm[ptr:ptr + bs]
                ptr += bs
            else:
                idx = np.arange(N)

            X_b, y_b = X[idx], y[idx]
            B = X_b.shape[0]

            scores = X_b @ self.weights                      # (B, C)
            true_score = scores[np.arange(B), y_b][:, None]  # (B, 1)
            margins = scores - true_score + 1.0              # (B, C)
            margins[np.arange(B), y_b] = 0.0                 # exclude true class

            # Crammer-Singer: only the worst violator contributes
            c_star = margins.argmax(axis=1)                  # (B,)
            worst  = margins[np.arange(B), c_star]
            active = worst > 0                               # margin violated

            sample_w = cw[y_b]                               # (B,)
            l2 = 0.5 * self.lambda_reg * np.sum(self.weights[:-1] ** 2)
            loss = (sample_w * np.maximum(worst, 0)).sum() / B + l2 / N
            self.loss_history.append(loss)

            grad = np.zeros_like(self.weights)
            if active.any():
                Xa  = X_b[active]
                ca  = c_star[active]
                ya  = y_b[active]
                wa  = sample_w[active][:, None]
                # +Xa to the violator column, -Xa to the true-class column
                np.add.at(grad.T, ca, (wa * Xa))             # (C, D+1) += (A, D+1)
                np.add.at(grad.T, ya, -(wa * Xa))
                grad /= B

            if self.lambda_reg > 0:
                grad[:-1] += (self.lambda_reg / N) * self.weights[:-1]

            lr_t = self.lr / (1.0 + self.lr_decay * t)
            velocity = self.beta * velocity + grad
            self.weights -= lr_t * velocity

        return self.predict(training_data)

    def decision_function(self, test_data):
        """Raw class scores w_c · x; argmax gives the prediction."""
        N = test_data.shape[0]
        X = np.hstack([test_data, np.ones((N, 1))])
        return X @ self.weights

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        return self.decision_function(test_data).argmax(axis=1)
