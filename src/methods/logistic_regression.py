import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, beta = 0.0, tol = None, init = 'zeros'):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None
        self.beta = beta 
        self.tol = tol
        self.init = init # 'zeros', 'random', 'xavier'
        self.loss_history = []

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """

        N,D = training_data.shape
        C = get_n_classes(training_labels)
        
        #Add bias column
        X = np.hstack([training_data, np.ones((N,1))]) #(N, D+1)
        Y = label_to_onehot(training_labels, C)  #(N,C)
        
        # Weight initialization
        input_dim = D + 1
        if self.init == 'zeros':
            self.weights = np.zeros((input_dim, C))
        elif self.init == 'random':
            self.weights = np.random.randn(input_dim, C)
        elif self.init == 'xavier':
            self.weights = np.random.randn(input_dim, C) * np.sqrt(2.0 / input_dim)
        
        velocity = np.zeros_like(self.weights)
        self.loss_history = []
        prev_loss = np.inf 
        
        for _ in range(self.max_iters): 
            #Softmax 
            logits = X @ self.weights 
            logits -= logits.max(axis=1, keepdims = True)
            exp = np.exp(logits)
            probs = exp/ exp.sum(axis=1, keepdims = True) #(N,C)
            
            #Cross Entropy loss
            loss = - np.mean(np.sum(Y * np.log(probs + 1e-15), axis = 1))
            self.loss_history.append(loss)
        
            #(Optional) Early stopping
            if self.tol is not None and abs(prev_loss - loss) < self.tol: 
                break 
            prev_loss = loss 
            
            #Gradient 
            grad = X.T @ (probs - Y) / N #(D+1, C)
            
            #(Optional) Momentum
            velocity = self.beta * velocity + (1 - self.beta) * grad
            self.weights -= self.lr * velocity
            
        # Final predictions with updated weights
        logits = X @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        
        pred_labels = onehot_to_label(probs)
            
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
            
        N = test_data.shape[0]
        X = np.hstack([test_data, np.ones((N, 1))])   # (N, D+1)
        logits = X @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        
        pred_labels = onehot_to_label(probs)

        return pred_labels