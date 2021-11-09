"""
FYS-STK4155 @UiO
Logistic regression.
"""
import numpy as np

from src.activations import sigmoid
from src.optimization import sgd
from src.utils import acc


sigmoid, _ = sigmoid()


class LogisticRegression:
    """Logistic Regression object for binary classification."""
    def fit(self, data, n_epochs, batch_size, lr, reg):
        """Fits the model given the data using SGD."""
        f = lambda beta, X, y: acc(X @ beta > 0., y)
        df = lambda beta, X, y: X.T @ (sigmoid(X @ beta) - y) / y.size + 2 * reg * beta
        self.beta, self.hist = sgd(data, f, df, n_epochs, batch_size, lr)
        return self

    def predict(self, X):
        """Returns the prediction for the given data."""
        return X @ self.beta > 0

    def score(self, X, y):
        """Stores the accuracy for the given data."""
        self.acc_test = acc(self.predict(X), y)
        return self.acc_test
