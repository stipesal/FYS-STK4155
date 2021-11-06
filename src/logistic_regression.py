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

        t.set_postfix(test_acc = acc(X_test @ beta > 0., y_test))

        hist["Train"].append(acc(X_train @ beta > 0., y_train))
        hist["Test"].append(acc(X_test @ beta > 0., y_test))
    return beta, hist


class LogisticRegression:
    def fit(self, data, n_epochs, batch_size, lr, reg):
        self.beta, self.hist = sgd(data, n_epochs, batch_size, lr, reg)
        return self

    def predict(self, X):
        return X @ self.beta > 0

    def score(self, X, y):
        self.acc_test = acc(self.predict(X), y)
        return self.acc_test
