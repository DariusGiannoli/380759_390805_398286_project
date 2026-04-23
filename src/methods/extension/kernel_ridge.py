"""
Kernelization (Lecture 7).

Implements the two estimators derived there:

* KernelRidgeRegression        — α = (K + λI)⁻¹ y,  ŷ(x) = k(X, x)ᵀ α
* KernelLeastSquaresClassifier — multi-output version on one-hot Y:
                                 α = (K + λI)⁻¹ Y,  ŷ(x) = argmax_c k(X,x)ᵀ α[:,c]

Three kernels, matching the lecture notation:
- 'linear' : k(x,y) = xᵀy
- 'poly'   : k(x,y) = (xᵀy + c)^d
- 'rbf'    : k(x,y) = exp(-‖x-y‖² / (2σ²))
"""

import numpy as np

from ...utils import get_n_classes, label_to_onehot, onehot_to_label


# --------------------------------------------------------------------------- #
# Kernel functions
# --------------------------------------------------------------------------- #

def _kernel_matrix(A, B, kernel='rbf', sigma=1.0, degree=2, coef0=1.0):
    if kernel == 'linear':
        return A @ B.T
    if kernel == 'poly':
        return (A @ B.T + coef0) ** degree
    if kernel == 'rbf':
        sqA = np.sum(A * A, axis=1, keepdims=True)          # (Na,1)
        sqB = np.sum(B * B, axis=1, keepdims=True).T        # (1,Nb)
        d2  = np.maximum(sqA + sqB - 2.0 * (A @ B.T), 0.0)
        return np.exp(-d2 / (2.0 * sigma ** 2))
    raise ValueError(f"unknown kernel: {kernel}")


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #

class KernelRidgeRegression(object):
    """
    Kernel ridge regression — α = (K + λI)⁻¹ (y − ȳ),  ŷ(x) = k(X,x)ᵀ α + ȳ.

    The lecture derivation gives α = (K + λI)⁻¹ y with no explicit intercept,
    but that pins predictions to pass through 0 in the feature space. Since
    our regression target has a non-zero mean (≈2.8), we center y before
    solving and add the mean back at predict time, which is the standard
    KRR implementation and restores equivalence with unbiased linear ridge.
    """

    def __init__(self, lambda_reg=1.0, kernel='rbf',
                 sigma=1.0, degree=2, coef0=1.0):
        """
        Arguments:
            lambda_reg (float): ridge strength added to the kernel matrix.
            kernel (str): 'linear', 'poly', or 'rbf'.
            sigma (float): RBF bandwidth (RBF kernel only).
            degree (int): polynomial degree (poly kernel only).
            coef0 (float): polynomial bias (poly kernel only).
        """
        self.lambda_reg = lambda_reg
        self.kernel     = kernel
        self.sigma      = sigma
        self.degree     = degree
        self.coef0      = coef0
        self.alpha      = None
        self.X_train    = None
        self.y_mean     = 0.0

    def _K(self, A, B):
        return _kernel_matrix(A, B, self.kernel,
                              self.sigma, self.degree, self.coef0)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D).
            training_labels (np.array): regression target of shape (N,).
        Returns:
            pred_labels (np.array): target of shape (N,).
        """
        self.X_train = training_data
        self.y_mean  = float(np.mean(training_labels))
        y_c = training_labels - self.y_mean

        N = training_data.shape[0]
        K = self._K(training_data, training_data)
        self.alpha = np.linalg.solve(K + self.lambda_reg * np.eye(N), y_c)
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D).
        Returns:
            pred_labels (np.array): labels of shape (N,).
        """
        return self._K(test_data, self.X_train) @ self.alpha + self.y_mean


# --------------------------------------------------------------------------- #
# Classification (multi-output KRR on one-hot labels)
# --------------------------------------------------------------------------- #

class KernelLeastSquaresClassifier(object):
    """
    Kernel least-squares classifier (Lecture 7, slides 43–52).

    Fits one kernel-ridge regressor per class on one-hot targets, then
    predicts argmax of the C scores. Completely faithful to the lecture
    derivation: W* = Φᵀ(K+λI)⁻¹ Y, dual form α = (K+λI)⁻¹ Y, ŷ = argmax_c
    (kernel-row · α)_c.
    """

    def __init__(self, lambda_reg=1.0, kernel='rbf',
                 sigma=1.0, degree=2, coef0=1.0):
        """
        Arguments:
            lambda_reg (float): ridge strength added to the kernel matrix.
            kernel (str): 'linear', 'poly', or 'rbf'.
            sigma (float): RBF bandwidth (RBF kernel only).
            degree (int): polynomial degree (poly kernel only).
            coef0 (float): polynomial bias (poly kernel only).
        """
        self.lambda_reg = lambda_reg
        self.kernel     = kernel
        self.sigma      = sigma
        self.degree     = degree
        self.coef0      = coef0
        self.alpha      = None        # (N, C)
        self.X_train    = None
        self.n_classes  = None

    def _K(self, A, B):
        return _kernel_matrix(A, B, self.kernel,
                              self.sigma, self.degree, self.coef0)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D).
            training_labels (np.array): class labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,).
        """
        self.X_train   = training_data
        self.n_classes = get_n_classes(training_labels)
        Y = label_to_onehot(training_labels, self.n_classes).astype(float)

        N = training_data.shape[0]
        K = self._K(training_data, training_data)
        self.alpha = np.linalg.solve(K + self.lambda_reg * np.eye(N), Y)
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D).
        Returns:
            pred_labels (np.array): labels of shape (N,).
        """
        scores = self._K(test_data, self.X_train) @ self.alpha      # (Nt, C)
        return onehot_to_label(scores)
