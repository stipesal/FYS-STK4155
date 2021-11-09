"""
FYS-STK4155 @UiO
Iterative optimization: (Mini-batch) stochastic gradient descent.
"""
import numpy as np
from tqdm import trange


def sgd(data, f, df, n_epochs, batch_size, lr, verbose=True):
    """Fits parameters using (mini-batch) stochastic gradient descent."""
    X_train, X_test, y_train, y_test = data

    n_batches = X_train.shape[0] // batch_size
    idx = np.arange(X_train.shape[0])

    beta = np.random.randn(X_train.shape[1])

    hist = {"Train": [], "Test": []}
    if verbose: t = trange(n_epochs, desc="Train")
    else: t = range(n_epochs)
    for _ in t:
        np.random.shuffle(idx)

        for b in range(n_batches):
            batch = idx[b * batch_size: (b + 1) * batch_size]
            beta -= lr * df(beta, X_train[batch], y_train[batch])

        hist["Train"].append(f(beta, X_train, y_train))
        hist["Test"].append(f(beta, X_test, y_test))
        train_score, test_score = hist["Train"][-1], hist["Test"][-1]
        if verbose:
            t.set_postfix(train=train_score, test=test_score)

    return beta, hist
