import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Lasso as Lasso_
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2


class LinearRegression():
    def __init__(self):
        pass

    def predict(self, X):
        return X @ self.beta

    def score(self, X, y):
        self.mse_test = mse(y, self.predict(X))
        self.r2_test = r2(y, self.predict(X))
        return self.mse_test


class OLS(LinearRegression):
    def __init__(self):
        pass

    def fit(self, X, y, confidence=None):
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        self.mse_train = mse(y, self.predict(X))
        self.r2_train = r2(y, self.predict(X))

        if confidence is not None:
            X_ = np.linalg.inv(X.T @ X)
            sigma = np.std(X @ self.beta, ddof=1)
            q = norm.ppf(confidence)
            dev = sigma * np.sqrt(np.diag(X_))
            self.CI = np.array([self.beta - q * dev, self.beta + q * dev]).T

        return self


class Ridge(LinearRegression):
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


class Lasso(LinearRegression):
    def __init__(self, reg_param):
        self.reg_param = reg_param
        self.model = Lasso_(alpha=self.reg_param)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.beta = self.model.coef_
        self.mse_train = mse(y, self.model.predict(X))
        self.r2_train = r2(y, self.model.predict(X))
        return self

    def predict(self, X):
        return self.model.predict(X)
