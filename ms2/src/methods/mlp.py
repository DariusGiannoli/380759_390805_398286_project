import numpy as np

from src.activations import Softmax
from src.losses import CrossEntropy


class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """
        assert len(activations) == len(dimensions) - 1, (
            "activations must have len(dimensions) - 1 entries"
        )
        self.dimensions = tuple(dimensions)
        self.activations = tuple(activations)
        self.L = len(dimensions) - 1  # number of weight layers

        # Xavier/Glorot initialization. Bias init to zeros.
        self.W = {}
        self.b = {}
        for l in range(1, self.L + 1):
            fan_in = dimensions[l - 1]
            fan_out = dimensions[l]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W[l] = np.random.randn(fan_in, fan_out) * scale
            self.b[l] = np.zeros(fan_out)

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors, shape (N, D).
        :return: (z, a)
                 z: dict with z[0] = x and z[l] = f(a[l]) for l >= 1
                 a: dict with a[l] = z[l-1] @ W[l] + b[l] for l >= 1
        """
        z = {0: x}
        a = {}
        for l in range(1, self.L + 1):
            a[l] = z[l - 1] @ self.W[l] + self.b[l]
            z[l] = self.activations[l - 1].forward(a[l])
        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters, shape (N, D).
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        z, _ = self.feed_forward(x)
        return z[self.L]

    def back_prop(self, z, a, y_true, loss):
        """
        Backpropagate the loss and return per-layer gradients.

        :param z: (dict) post-activations from feed_forward (z[0] is input)
        :param a: (dict) pre-activations from feed_forward
        :param y_true: (array) one-hot/target, shape (N, C)
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return: (dW, db) dicts indexed by layer.
        """
        dW = {}
        db = {}

        # Output layer. Softmax has a dense Jacobian, so handle its
        # vector-Jacobian product explicitly instead of pretending it is an
        # elementwise activation. For Softmax+CE this reduces to the usual
        # closed form (p - y) / N.
        output_activation = self.activations[self.L - 1]
        if output_activation is Softmax:
            if loss is CrossEntropy:
                delta = loss.gradient(y_true, z[self.L])
            else:
                grad_pred = loss.gradient(y_true, z[self.L])
                delta = z[self.L] * (
                    grad_pred - np.sum(grad_pred * z[self.L], axis=1, keepdims=True)
                )
        else:
            delta = loss.gradient(y_true, z[self.L]) * output_activation.gradient(a[self.L])

        for l in range(self.L, 0, -1):
            dW[l] = z[l - 1].T @ delta
            db[l] = np.sum(delta, axis=0)
            if l > 1:
                # Propagate the error backward through W[l] then apply the
                # activation derivative of the previous layer.
                delta = (delta @ self.W[l].T) * self.activations[l - 2].gradient(a[l - 1])

        return dW, db

    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases of a single layer with optional heavy-ball
        momentum. Velocities are persisted on the object so successive calls
        accumulate them across mini-batches.

        :param index: (int) layer number (1..L)
        :param dw: (array) gradient w.r.t. W[index]
        :param delta: (array) gradient w.r.t. b[index]
        """
        if self.beta > 0.0:
            self.vW[index] = self.beta * self.vW[index] + dw
            self.vb[index] = self.beta * self.vb[index] + delta
            self.W[index] -= self.lr * self.vW[index]
            self.b[index] -= self.lr * self.vb[index]
        else:
            self.W[index] -= self.lr * dw
            self.b[index] -= self.lr * delta

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3,
            beta=0.0, track_loss=False):
        """
        Train the model with mini-batch SGD, optionally with heavy-ball
        momentum.

        :param x: (array) inputs, shape (N, D)
        :param y_true: (array) one-hot/target labels, shape (N, C)
        :param loss: Loss class (MSE, CrossEntropy, ...)
        :param epochs: (int) number of full passes over the data
        :param batch_size: (int) mini-batch size
        :param learning_rate: (float) SGD step size
        :param beta: (float) momentum coefficient (0.0 = vanilla SGD)
        :param track_loss: (bool) if True, store per-epoch training loss in
                           self.loss_history (cheap on this dataset).
        """
        self.lr = learning_rate
        self.beta = float(beta)
        self.vW = {l: np.zeros_like(self.W[l]) for l in range(1, self.L + 1)}
        self.vb = {l: np.zeros_like(self.b[l]) for l in range(1, self.L + 1)}
        self.loss_history = []

        N = x.shape[0]
        bs = max(1, min(int(batch_size), N))

        for _ in range(int(epochs)):
            perm = np.random.permutation(N)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                x_batch = x[idx]
                y_batch = y_true[idx]

                z, a = self.feed_forward(x_batch)
                dW, db = self.back_prop(z, a, y_batch, loss)
                for l in range(1, self.L + 1):
                    self.update_w_b(l, dW[l], db[l])

            if track_loss:
                self.loss_history.append(loss.loss(y_true, self.predict(x)))

        return self.predict(x)
