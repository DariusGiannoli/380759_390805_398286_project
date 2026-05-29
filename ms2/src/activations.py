import numpy as np


class Sigmoid:
    @staticmethod
    def forward(z):
        # Clip to avoid overflow in exp(-z) for large negative z.
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def gradient(z):
        s = Sigmoid.forward(z)
        return s * (1.0 - s)


class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0.0, z)

    @staticmethod
    def gradient(z):
        return (z > 0).astype(z.dtype)


class Identity:
    """Linear activation. Useful as the output of a regression network."""

    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z)


class Softmax:
    """
    Row-wise softmax. Its derivative is a dense per-sample Jacobian, so
    MLP.back_prop handles Softmax output layers explicitly. The gradient
    method keeps the same simple activation interface as the other classes.
    """

    @staticmethod
    def forward(z):
        z_shift = z - np.max(z, axis=-1, keepdims=True)
        exp = np.exp(z_shift)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def gradient(z):
        return np.ones_like(z)
