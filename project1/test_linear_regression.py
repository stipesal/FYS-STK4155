"""
FYS-STK4155 @UiO, PROJECT I.
Testing: Linear Regression.
"""
import pytest
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as SKL_Ridge

from linear_regression import OLS, Ridge


np.random.seed(2021)


@pytest.fixture(scope="session")
def data():
    """Regression dummy data."""
    X, y = make_regression(n_samples=1000, n_features=100)
    return X, y


def test_OLS(data):
    """
    Tests the ordinary least squares method by comparing
    the regression coefficients to Scikit-Learn's.
    """
    model = OLS().fit(*data)
    skl_model = LinearRegression().fit(*data)

    assert np.allclose(model.beta, skl_model.coef_)


def test_Ridge(data):
    """
    Tests the Ridge regression by comparing
    the regression coefficients to Scikit-Learn's.
    """
    lmbd = 1E-5
    model = Ridge(reg_param=lmbd).fit(*data)
    skl_model = SKL_Ridge(alpha=lmbd).fit(*data)

    assert np.allclose(model.beta, skl_model.coef_)
