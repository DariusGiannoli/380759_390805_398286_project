import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification", weighted=False, metric='l2'):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.weighted = weighted  # False = uniform vote, True = distance-weighted
        self.metric = metric      # 'l2', 'l1', 'cosine'

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        return self.predict(training_data)

    def _compute_distances(self, test_data):
        """Returns (N_test, N_train) distance matrix."""
        if self.metric == 'l2':
            sq_test  = np.sum(test_data ** 2, axis=1, keepdims=True)
            sq_train = np.sum(self.training_data ** 2, axis=1)
            cross    = test_data @ self.training_data.T
            return np.sqrt(np.maximum(sq_test + sq_train - 2 * cross, 0))

        elif self.metric == 'l1':
            return np.sum(np.abs(test_data[:, None, :] - self.training_data[None, :, :]), axis=2)

        elif self.metric == 'cosine':
            t  = test_data / (np.linalg.norm(test_data, axis=1, keepdims=True) + 1e-10)
            tr = self.training_data / (np.linalg.norm(self.training_data, axis=1, keepdims=True) + 1e-10)
            return 1 - t @ tr.T

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        distances = self._compute_distances(test_data)           # (N_test, N_train)
        k_idx     = np.argsort(distances, axis=1)[:, :self.k]    # (N_test, k)
        k_labels  = self.training_labels[k_idx]                   # (N_test, k)
        k_dists   = np.take_along_axis(distances, k_idx, axis=1)  # (N_test, k)

        if self.weighted:
            weights = 1.0 / (k_dists + 1e-10)
        else:
            weights = np.ones_like(k_dists)

        if self.task_kind == "classification":
            classes     = np.unique(self.training_labels).astype(int)
            votes       = np.array([
                (weights * (k_labels == c)).sum(axis=1) for c in classes]).T                                                  # (N_test, C)
            pred_labels = classes[np.argmax(votes, axis=1)]

        else:  # regression
            pred_labels = np.sum(weights * k_labels, axis=1) / np.sum(weights, axis=1)

        return pred_labels
