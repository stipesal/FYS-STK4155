import matplotlib.pyplot as plt
from numba import jit
import numpy as np

from scipy.special import binom
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def plot_3d():
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    return ax

@jit(nopython=True)
def design_matrix(data, degree):
    """Compare to sklearns' PolynomialFeatures."""
    monoms = []
    for n in range(degree + 1):
        for k in range(n + 1):
            monoms.append((n - k, k))

    x, y = data.T
    X = np.zeros((x.shape[0], len(monoms)))
    for ix, (i, j) in enumerate(monoms):
        X[:, ix] = x ** i * y ** j

    return X


def bias_variance_analysis(model, X_train, X_test, y_train, y_test, n_bootstraps):
    y_pred = np.zeros((n_bootstraps, y_test.shape[0]))

    for i in range(n_bootstraps):
        x_, y_ = resample(X_train, y_train)
        y_pred[i] = model.fit(x_, y_).predict(X_test)
    
    error = np.mean((y_pred - y_test) ** 2)
    bias = np.mean((y_test - np.mean(y_pred, axis=0)) ** 2)
    var = np.mean(np.var(y_pred, axis=0))

    return error, bias, var


def bootstrap(model, X, y, n_bootstraps):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.2)

    model.boot = {
        "Train MSE": [],
        "Test MSE": [],
        "beta": np.zeros((n_bootstraps, X_train.shape[1])),
    }
    for i in range(n_bootstraps):
        x_, y_ = resample(X_train, Y_train)
        model.fit(x_, y_)
        model.score(X_test, Y_test)

        model.boot["Train MSE"].append(model.mse_train)
        model.boot["Test MSE"].append(model.mse_test)
        model.boot["beta"][i] = model.beta


def cross_validation(model, X, y, n_folds):
    idx = np.arange(X.shape[0])
    size = X.shape[0] // n_folds

    model.cv = {
        "Train MSE": [],
        "Test MSE": [],
    }
    for i in range(n_folds):
        train = idx[i * size : (i+1) * size]
        test = np.hstack((idx[:i * size], idx[(i+1) * size :]))

        model.fit(X[train], y[train])

        model.cv["Train MSE"].append(model.score(X[train], y[train]))
        model.cv["Test MSE"].append(model.score(X[test], y[test]))
