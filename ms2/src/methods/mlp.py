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

    def feed_forward(self, x, training=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors, shape (N, D).
        :param training: (bool) If True and self.dropout_p > 0, apply inverted
                         dropout to hidden activations (not the output layer).
                         The sampled masks are stashed on self so back_prop
                         can apply the same mask on the backward pass.
        :return: (z, a)
                 z: dict with z[0] = x and z[l] = f(a[l]) for l >= 1.
                 When dropout is active, z[l] stores the masked activation,
                 which is also what feeds into the next layer.
                 a: dict with a[l] = z[l-1] @ W[l] + b[l] for l >= 1
        """
        z = {0: x}
        a = {}
        do_dropout = training and getattr(self, 'dropout_p', 0.0) > 0.0
        self._dropout_masks = {} if do_dropout else None
        for l in range(1, self.L + 1):
            a[l] = z[l - 1] @ self.W[l] + self.b[l]
            z[l] = self.activations[l - 1].forward(a[l])
            # Dropout on hidden layers only (never on the output).
            if do_dropout and l < self.L:
                keep = 1.0 - self.dropout_p
                mask = (np.random.rand(*z[l].shape) < keep).astype(z[l].dtype) / keep
                z[l] = z[l] * mask
                self._dropout_masks[l] = mask
        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters, shape (N, D).
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        z, _ = self.feed_forward(x, training=False)
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

        masks = getattr(self, '_dropout_masks', None)
        for l in range(self.L, 0, -1):
            dW[l] = z[l - 1].T @ delta
            db[l] = np.sum(delta, axis=0)
            if l > 1:
                # Backward through W[l] (gives d/d(masked z[l-1])).
                grad_z = delta @ self.W[l].T
                # If dropout was applied to z[l-1] on the forward pass, the
                # same mask must scale the gradient flowing back into a[l-1].
                if masks is not None and (l - 1) in masks:
                    grad_z = grad_z * masks[l - 1]
                delta = grad_z * self.activations[l - 2].gradient(a[l - 1])

        return dW, db

    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases of a single layer with optional heavy-ball
        momentum and L2 weight decay. Velocities are persisted on the object
        so successive calls accumulate them across mini-batches.

        Weight decay is added to the weight gradient only (biases are not
        decayed, which is the standard convention).

        :param index: (int) layer number (1..L)
        :param dw: (array) gradient w.r.t. W[index]
        :param delta: (array) gradient w.r.t. b[index]
        """
        if self.weight_decay > 0.0:
            dw = dw + self.weight_decay * self.W[index]
        if self.beta > 0.0:
            self.vW[index] = self.beta * self.vW[index] + dw
            self.vb[index] = self.beta * self.vb[index] + delta
            self.W[index] -= self.lr * self.vW[index]
            self.b[index] -= self.lr * self.vb[index]
        else:
            self.W[index] -= self.lr * dw
            self.b[index] -= self.lr * delta

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3,
            beta=0.0, weight_decay=0.0, dropout=0.0,
            x_val=None, y_val=None, patience=None,
            track_loss=False):
        """
        Train the model with mini-batch SGD, optionally with heavy-ball
        momentum, L2 weight decay, dropout on hidden activations, and
        early stopping on a validation loss.

        :param x: (array) inputs, shape (N, D)
        :param y_true: (array) one-hot/target labels, shape (N, C)
        :param loss: Loss class (MSE, CrossEntropy, ...)
        :param epochs: (int) maximum number of full passes over the data
        :param batch_size: (int) mini-batch size
        :param learning_rate: (float) SGD step size
        :param beta: (float) momentum coefficient (0.0 = vanilla SGD)
        :param weight_decay: (float) L2 coefficient added to W-gradient
        :param dropout: (float) probability of dropping a hidden unit
                        (0.0 disables dropout)
        :param x_val, y_val: (array, optional) validation set; required when
                             ``patience`` is set.
        :param patience: (int, optional) early-stopping patience. If set,
                         training halts after ``patience`` consecutive epochs
                         without a strict improvement in validation loss,
                         and the lowest-val-loss weights are restored.
                         Records the stopping epoch in ``self.stopped_epoch_``.
        :param track_loss: (bool) if True, store per-epoch training loss in
                           self.loss_history (cheap on this dataset).
        """
        self.lr = learning_rate
        self.beta = float(beta)
        self.weight_decay = float(weight_decay)
        self.dropout_p = float(dropout)
        self.vW = {l: np.zeros_like(self.W[l]) for l in range(1, self.L + 1)}
        self.vb = {l: np.zeros_like(self.b[l]) for l in range(1, self.L + 1)}
        self.loss_history = []
        self.val_loss_history = []
        self.stopped_epoch_ = None

        do_early_stop = patience is not None and x_val is not None and y_val is not None
        best_val_loss = np.inf
        best_W, best_b = None, None
        bad_epochs = 0

        N = x.shape[0]
        bs = max(1, min(int(batch_size), N))

        for epoch in range(int(epochs)):
            perm = np.random.permutation(N)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                x_batch = x[idx]
                y_batch = y_true[idx]

                z, a = self.feed_forward(x_batch, training=True)
                dW, db = self.back_prop(z, a, y_batch, loss)
                for l in range(1, self.L + 1):
                    self.update_w_b(l, dW[l], db[l])

            if track_loss:
                self.loss_history.append(loss.loss(y_true, self.predict(x)))

            if do_early_stop:
                val_loss = loss.loss(y_val, self.predict(x_val))
                self.val_loss_history.append(val_loss)
                if val_loss < best_val_loss - 1e-8:
                    best_val_loss = val_loss
                    best_W = {l: self.W[l].copy() for l in self.W}
                    best_b = {l: self.b[l].copy() for l in self.b}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(patience):
                        self.stopped_epoch_ = epoch + 1
                        break

        if do_early_stop and best_W is not None:
            self.W = best_W
            self.b = best_b

        return self.predict(x)
