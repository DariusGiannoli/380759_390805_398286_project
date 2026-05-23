import numpy as np
from itertools import combinations_with_replacement


class LinearRegression(object):
    """
    Ridge-regularized linear regression with optional polynomial feature
    expansion. The bias term is never regularized.
    """

    def __init__(self, lambda_reg=0.0, method='closed_form', lr=0.01,
                 max_iters=1000, degree=1, interaction=False):
        self.lambda_reg = lambda_reg    # L2 regularization strength
        self.method = method            # 'closed_form' or 'gradient_descent'
        self.lr = lr                    # learning rate for gradient descent
        self.max_iters = max_iters
        self.degree = degree            # 1 = standard, 2+ = polynomial expansion
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

        D = X.shape[1]
        features = [X]

        if self.interaction:
            # Full polynomial: all combinations with replacement up to degree
            for d in range(2, self.degree + 1):
                for combo in combinations_with_replacement(range(D), d):
                    features.append(np.prod(X[:, combo], axis=1, keepdims=True))
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
        N = training_data.shape[0]
        X_poly = self._expand_features(training_data)
        X = np.hstack([X_poly, np.ones((N, 1))])  # last column = bias

        if self.method == 'closed_form':
            self._fit_closed_form(X, training_labels)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, training_labels)
        else:
            raise ValueError(
                "method must be 'closed_form' or 'gradient_descent'"
            )

        return X @ self.weights

    def _fit_closed_form(self, X, y):
        """Closed-form solution:  w = (XᵀX + λI)⁻¹ Xᵀy  (bias row unregularized)."""
        reg = np.eye(X.shape[1])
        reg[-1, -1] = 0.0
        self.weights = np.linalg.solve(X.T @ X + self.lambda_reg * reg, X.T @ y)

    def _fit_gradient_descent(self, X, y):
        """
        Iterative solution via gradient descent. Minimizes the same objective
        as the closed form, ‖Xw − y‖² + λ‖w[:-1]‖², but scaled by 1/N so a
        moderate learning rate (~1e-2) works for any N. Zeroing the gradient
        gives (XᵀX + λI)w = Xᵀy, matching the closed-form solution.
        """
        N = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.loss_history = []

        for _ in range(self.max_iters):
            residuals = X @ self.weights - y
            # (scaled) objective at current weights: MSE + (λ/N)·‖w[:-1]‖²
            mse = np.mean(residuals ** 2)
            l2  = (self.lambda_reg / N) * np.sum(self.weights[:-1] ** 2)
            self.loss_history.append(mse + l2)
            grad = (2.0 / N) * X.T @ residuals
            grad[:-1] += (2.0 * self.lambda_reg / N) * self.weights[:-1]
            self.weights -= self.lr * grad

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