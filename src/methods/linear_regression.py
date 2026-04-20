import numpy as np
from itertools import combinations_with_replacement


class LinearRegression(object):
    """
    Linear regression with optional polynomial feature expansion.
    """

    def __init__(self, lambda_reg=0, method='closed_form', lr=0.01, max_iters=1000,
                 degree=1, interaction=False):
        self.lambda_reg = lambda_reg  # L2 regularization strength
        self.method = method          # 'closed_form' or 'gradient_descent'
        self.lr = lr
        self.max_iters = max_iters
        self.degree = degree          # 1 = standard, 2+ = polynomial expansion
        self.interaction = interaction  # False = per-feature powers, True = full cross-terms
        self.weights = None
        self.loss_history = []

    def _expand_features(self, X):
        """
        Polynomial feature expansion.
        degree=1 : returns X unchanged
        degree>1, interaction=False : adds x^2, x^3, ... per feature (no cross-terms)
        degree>1, interaction=True  : full expansion including cross-terms (x1*x2, etc.)
        """
        if self.degree == 1:
            return X

        N, D = X.shape
        features = [X]

        if self.interaction:
            # Full polynomial: all combinations with replacement up to degree
            for d in range(2, self.degree + 1):
                for combo in combinations_with_replacement(range(D), d):
                    new_feat = np.prod(X[:, combo], axis=1, keepdims=True)
                    features.append(new_feat)
        else:
            # Per-feature only: x^2, x^3, ... no cross-terms
            for d in range(2, self.degree + 1):
                features.append(X ** d)

        return np.hstack(features)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        N, D = training_data.shape
        X_poly = self._expand_features(training_data)
        D_poly = X_poly.shape[1]
        X = np.hstack([X_poly, np.ones((N, 1))])

        if self.method == 'closed_form':
            self._fit_closed_form(X, training_labels, D_poly)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, training_labels, N, D_poly)

        return X @ self.weights

    def _fit_closed_form(self, X, training_labels, D):
        """Closed-form solution: w = (XᵀX + λI)⁻¹ Xᵀy"""
        I = np.eye(D + 1)
        I[-1, -1] = 0  # don't regularize bias
        self.weights = np.linalg.solve(X.T @ X + self.lambda_reg * I, X.T @ training_labels)

    def _fit_gradient_descent(self, X, training_labels, N, D):
        """Iterative solution via gradient descent."""
        self.weights = np.zeros(D + 1)
        self.loss_history = []

        for _ in range(self.max_iters):
            preds = X @ self.weights
            residuals = preds - training_labels
            grad = (2 / N) * X.T @ residuals
            grad[:-1] += 2 * self.lambda_reg * self.weights[:-1]  # don't regularize bias
            self.weights -= self.lr * grad
            self.loss_history.append(np.mean(residuals ** 2))

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        N = test_data.shape[0]
        X_poly = self._expand_features(test_data)
        X = np.hstack([X_poly, np.ones((N, 1))])
        return X @ self.weights
