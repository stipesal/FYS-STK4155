"""
FYS-STK4155 @UiO
Logistic regression.
"""
import numpy as np

from src.activations import sigmoid
from src.utils import acc
from tqdm import tqdm


sigmoid, _ = sigmoid()


def sgd(data, n_epochs, batch_size, lr, reg):
    """Fits parameters using (mini-batch) stochastic gradient descent."""
    X_train, X_test, y_train, y_test = data

    n_batches = X_train.shape[0] // batch_size
    idx = np.arange(X_train.shape[0])

    beta = np.random.randn(X_train.shape[1])

    def df(beta, X, y):
        pred = sigmoid(X @ beta)
        return X.T @ (pred - y) / y.size + 2 * reg * beta

    hist = {"Train": [], "Test": []}
    t = tqdm(range(n_epochs))
    for _ in t:
        np.random.shuffle(idx)
        for b in range(n_batches):
            batch = idx[b * batch_size: (b + 1) * batch_size]
            beta -= lr * df(beta, X_train[batch], y_train[batch])

        train_score = acc(X_train @ beta > 0., y_train)
        test_score = acc(X_test @ beta > 0., y_test)
        hist["Train"].append(train_score)
        hist["Test"].append(test_score)

        t.set_postfix(test=test_score)

    return beta, hist


class LogisticRegression:
    """Logistic Regression object for binary classification."""
    def fit(self, data, n_epochs, batch_size, lr, reg):
        """Fits the model given the data using SGD."""
        self.beta, self.hist = sgd(data, n_epochs, batch_size, lr, reg)
        return self

    def predict(self, X):
        """Returns the prediction for the given data."""
        return X @ self.beta > 0

    def score(self, X, y):
        """Stores the accuracy for the given data."""
        self.acc_test = acc(self.predict(X), y)
        return self.acc_test
