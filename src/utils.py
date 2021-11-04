"""
FYS-STK4155 @UiO
Basic, useful functions.
"""
import matplotlib.pyplot as plt
from numba import jit
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


LEGEND_SIZE = 12
LABEL_SIZE = 12


def plot_3d():
    """Returns an 3D axis object with labeled axis."""
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r"$x$", size=LABEL_SIZE)
    ax.set_ylabel(r"$y$", size=LABEL_SIZE)
    ax.set_zlabel(r"$z$", size=LABEL_SIZE)
    return ax


@jit(nopython=True)  # Compile just-in-time.
def design_matrix(data, degree):
    """Assembles the design matrix needed for linear regression."""
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
    """
    Estimates and returns the error, bias, and variance
    of the model given the data by using the bootstrap method.
    """
    y_pred = np.zeros((n_bootstraps, y_test.shape[0]))

    for i in range(n_bootstraps):
        x_, y_ = resample(X_train, y_train)
        y_pred[i] = model.fit(x_, y_).predict(X_test)
    
    error = np.mean((y_pred - y_test) ** 2)
    bias = np.mean((y_test - np.mean(y_pred, axis=0)) ** 2)
    var = np.mean(np.var(y_pred, axis=0))

    return error, bias, var


def bootstrap(model, X, y, n_bootstraps):
    """Fits the model given the data using the bootstrap method."""
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.2)

    model.boot = {
        "Train MSE": [],
        "Test MSE": [],
        "beta": np.zeros((n_bootstraps, X_train.shape[1])),
    }
    for i in range(n_bootstraps):
        x_, y_ = resample(X_train, Y_train)

        model.fit(x_, y_)

        model.boot["Train MSE"].append(model.score(x_, y_))
        model.boot["Test MSE"].append(model.score(X_test, Y_test))
        model.boot["beta"][i] = model.beta


def cross_validation(model, X, y, n_folds):
    """Fits the model given the data using cross validation."""
    idx = np.arange(X.shape[0])
    size = X.shape[0] // n_folds

    model.cv = {
        "Train MSE": [],
        "Test MSE": [],
    }
    for i in range(n_folds):
        test= idx[i * size : (i+1) * size]
        train = np.hstack((idx[:i * size], idx[(i+1) * size :]))

        model.fit(X[train], y[train])

        model.cv["Train MSE"].append(model.score(X[train], y[train]))
        model.cv["Test MSE"].append(model.score(X[test], y[test]))


def mse(y_pred, y_true):
    """Returns the mean-squared error between predictions and true values."""
    return ((y_pred - y_true) ** 2).sum() / y_true.size


def r2(y_pred, y_true):
    """Returns the R2 score between predictions and true value."""
    u = ((y_true - y_pred)** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1. - u / v


def acc(y_pred, y_true):
    """Returns the accuracy between predictions and true labels."""
    return (y_pred == y_true).mean()


def ohe(labels, n_classes):
    """Returns a one-hot encoded array for the given labels."""
    ohe = np.zeros((labels.size, n_classes))
    ohe[np.arange(labels.size), labels] = 1.
    return ohe


def scale(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler
