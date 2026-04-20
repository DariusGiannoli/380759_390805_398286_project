import numpy as np


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self, lambda_reg=0, method = 'closed_form', lr = 0.01, max_iters = 1000):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.lambda_reg = lambda_reg # regularization parameter 
        self.method = method # method to solve linear regression, either 'closed_form' or 'gradient_descent'
        self.lr = lr # learning rate for gradient descent
        self.max_iters = max_iters # maximum number of iterations for gradient descent
        self.weights = None # weights learned during fit()
        self.loss_history = [] # to store the loss history during gradient descent
        
        
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        N, D = training_data.shape
        X = np.hstack([training_data, np.ones((N, 1))])

        if self.method == 'closed_form':
            self._fit_closed_form(X, training_labels, D)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, training_labels, N, D)

        return X @ self.weights 
    
    def _fit_closed_form(self, X, training_labels, D):
        """
        Private helper method to fit the model using closed-form solution
        Closed-form solution: w = (XᵀX + λI)⁻¹ Xᵀy
        """
        I = np.eye(D + 1)
        I[-1, -1] = 0  # don't regularize bias
        XtX = X.T @ X
        Xty = X.T @ training_labels
        self.weights = np.linalg.solve(XtX + self.lambda_reg * I, Xty) 
        
    def _fit_gradient_descent(self, X, training_labels, N, D):
        """
        Private helper method to fit the model using gradient descent
        Iterative solution via gradient descent
        """
        self.weights = np.zeros(D + 1)
        self.loss_history = []

        for _ in range(self.max_iters):
            preds = X @ self.weights
            residuals = preds - training_labels           # (N,)
            # gradient of MSE loss + L2 regularization
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
        X = np.hstack([test_data, np.ones((N, 1))])
        return X @ self.weights
