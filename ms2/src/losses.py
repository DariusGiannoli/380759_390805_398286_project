import numpy as np


class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        Mean squared error averaged over all elements.
        :param y_true: (array) target, shape (N, C)
        :param y_pred: (array) prediction, shape (N, C)
        :return: (float) scalar loss
        """
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Gradient w.r.t. y_pred. Scaled by 1/N so gradients are batch-averaged
        and the learning rate is decoupled from batch size.
        """
        N = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / N


class CrossEntropy:
    """
    Categorical cross-entropy loss. Expects y_pred to be softmax probabilities
    and y_true to be one-hot. .gradient() returns the analytical combined
    Softmax+CE delta, (probs - y_true)/N, which MLP.back_prop uses directly
    when the output activation is Softmax.
    """

    @staticmethod
    def loss(y_true, y_pred):
        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(np.sum(y_true * np.log(y_pred_safe), axis=1)))

    @staticmethod
    def gradient(y_true, y_pred):
        N = y_true.shape[0]
        return (y_pred - y_true) / N
