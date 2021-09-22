import matplotlib.pyplot as plt
import numpy as np

from scipy.special import binom
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.utils import resample


def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9*x - 2)**2) - 0.25 * ((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2) / 49.0 - 0.1 * (9*y + 1))
    term3 = 0.50 * np.exp(-(9*x - 7)**2 / 4.0 - 0.25 * ((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4


def plot_franke_function(ax):
    x = np.linspace(0, 1, 50)
    x, y = np.meshgrid(x, x)
    ax.plot_wireframe(
        x, y, franke_function(x, y),
        color="k",
        label="Franke's function",
        alpha=.3,
    )


def plot_3d():
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    return ax


def sample_franke_function(N, noise):
    """
    Samples `N` uniform data points in the unit square `[0,1)^2`,
    as well as the corresponding evaluation in Franke's function.
    """
    X = np.random.rand(N, 2)
    x, y = X.T

    Y = franke_function(x, y)
    Y += noise * np.random.randn(N)
    return X, Y


def design_matrix(data, degree):
    """Compare to sklearns' PolynomialFeatures."""
    monoms = []
    for n in range(degree + 1):
        for k in range(n + 1):
            monoms.append((n - k, k))

    # The number of monoms (or features) is equal to:
    # nCr(degree + 2, degree) or (degree + 1) * (degree + 2) / 2
    assert len(monoms) == binom(degree + 2, degree)

    x, y = data.T
    X = np.zeros((x.shape[0], len(monoms)))
    for ix, (i, j) in enumerate(monoms):
        X[:, ix] = x ** i * y ** j

    return X


class LinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y, confidence=None):
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        self.mse_train = mse(y, self.predict(X))
        self.r2_train = r2(y, self.predict(X))

        if confidence is not None:
            X_ = np.linalg.inv(X.T @ X)
            sigma = np.std(X @ self.beta, ddof=1)    # ddof: 1/(n-1) instead of 1/n.
            q = norm.ppf(confidence)
            dev = np.sqrt(sigma**2 * np.diag(X_))
            self.CI = np.array([self.beta - q * dev, self.beta + q * dev]).T

        return self
    
    def predict(self, X):
        return X @ self.beta
    
    def score(self, X, y):
        self.mse_test = mse(y, self.predict(X))
        self.r2_test = r2(y, self.predict(X))
        return self.mse_test


class RidgeRegression():
    def __init__(self, reg_param):
        self.reg_param = reg_param
    
    def fit(self, X, y):
        I = np.eye(X.shape[1])
        self.beta = np.linalg.solve(
            X.T @ X + self.reg_param * I,
            X.T @ y,
        )
        self.mse_train = mse(y, self.predict(X))
        self.r2_train = r2(y, self.predict(X))
        return self
    
    def predict(self, X):
        return X @ self.beta

    def score(self, X, y):
        self.mse_test = mse(y, self.predict(X))
        self.r2_test = r2(y, self.predict(X))
        return self.mse_test


class LassoRegression():
    def __init__(self, reg_param):
        self.reg_param = reg_param
        self.model = Lasso(alpha=self.reg_param)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        # self.beta = np.concatenate(
        #     (
        #     self.model.intercept_,
        #     self.model.coef_,
        #     )
        # )
        self.beta = self.model.coef_
        self.mse_train = mse(y, self.model.predict(X))
        self.r2_train = r2(y, self.model.predict(X))
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        self.mse_test = mse(y, self.model.predict(X))
        self.r2_test = r2(y, self.model.predict(X))
        return self.mse_test


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
    }
    for _ in range(n_bootstraps):
        x_, y_ = resample(X_train, Y_train)
        model.fit(x_, y_)
        model.score(X_test, Y_test)

        model.boot["Train MSE"].append(model.mse_train)
        model.boot["Test MSE"].append(model.mse_test)


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
