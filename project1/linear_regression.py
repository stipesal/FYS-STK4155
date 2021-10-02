"""
FYS-STK4155 @UiO, PROJECT I.
Linear Regression: OLS, Ridge and Lasso regression.
"""
import numpy as np
from scipy.stats import t
from sklearn.linear_model import Lasso as Lasso_
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2


class LinearRegression():
    """Base Linear Regression object with the basic methods."""
    def __init__(self):
        """Sets subclass-specific parameters."""
        pass

    def fit(self, X, y):
        """Fits the model given the data. See subclasses."""
        pass

    def predict(self, X):
        """Returns the prediction for the given data."""
        return X @ self.beta

    def score(self, X, y):
        """Stores the MSE and R2 for the given data."""
        self.mse_test = mse(y, self.predict(X))
        self.r2_test = r2(y, self.predict(X))
        return self.mse_test


class OLS(LinearRegression):
    def fit(self, X, y, confidence=None):
        """
        Confidence should be in `(0, 1)`, e.g. 'confidence=0.95'
        for a 95% confidence interval for the parameters.
        """
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        self.mse_train = mse(y, self.predict(X))
        self.r2_train = r2(y, self.predict(X))

        if confidence is not None:
            sigma = ((X @ self.beta - y) ** 2).sum() / (y.size - 2)
            X_ = np.linalg.inv(X.T @ X)
            q = t.ppf(1 - (1 - confidence) / 2, y.size - 2)
            dev = np.sqrt(sigma * np.diag(X_))
            self.CI = np.array([self.beta - q * dev, self.beta + q * dev]).T

        return self


class Ridge(LinearRegression):
    def __init__(self, reg_param):
        """Sets the regularization parameter for the penalty term."""
        self.reg_param = reg_param

    def fit(self, X, y):
        """See base class."""
        I = np.eye(X.shape[1])
        self.beta = np.linalg.solve(
            X.T @ X + self.reg_param * I,
            X.T @ y,
        )
        self.mse_train = mse(y, self.predict(X))
        self.r2_train = r2(y, self.predict(X))
        return self


class Lasso(LinearRegression):
    """Basically Scikit-Learn's Lasso."""
    def __init__(self, reg_param):
        """
        Sets the regularization parameter for the penalty term,
        as well as Scikit-Learn's Lasso model.
        """
        self.reg_param = reg_param
        self.model = Lasso_(alpha=self.reg_param)

    def fit(self, X, y):
        """See base class."""
        self.model.fit(X, y)
        self.beta = self.model.coef_
        self.mse_train = mse(y, self.model.predict(X))
        self.r2_train = r2(y, self.model.predict(X))
        return self

    def predict(self, X):
        """See base class."""
        return self.model.predict(X)
