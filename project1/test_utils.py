import pytest
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from utils import design_matrix, bias_variance_analysis, cross_validation
from linear_regression import OLS


np.random.seed(2021)


@pytest.fixture(scope="session")
def data():
    X, y = make_regression(n_samples=1000, n_features=100)
    return X, y


def test_design_matrix():
    """
    Checks the number of features (columns) and
    compares with Scikit-Learn's PolynomialFeatures.
    """
    N = 100
    degree = 10
    data = np.random.rand(N, 2)

    X = design_matrix(data, degree)
    poly = PolynomialFeatures(degree=degree).fit_transform(data)

	# Number of monoms (or features): (degree + 1) * (degree + 2) / 2
    assert X.shape[1] == (degree + 1) * (degree + 2) / 2
    assert np.allclose(X, poly)


def test_bias_variance_analysis(data):
    """
    Checks the theoretical error composition
    of the bias-variance tradeoff.
    """
    model = OLS()
    n_bootstraps = 50

    err, bias, var = bias_variance_analysis(
        model, *train_test_split(*data, test_size=.2), n_bootstraps
    )
    # Error composition.
    assert np.allclose(err, bias + var)


def test_cross_validation(data):
    """
    Tests the cross-validation method by
    comparing it to Scikit-Learn's.
    """
    n_folds = 5

    model = OLS()
    cross_validation(model, *data, n_folds)

    skl_model = LinearRegression()
    skl_cv = cross_validate(skl_model, *data, cv=n_folds, scoring='neg_mean_squared_error')

    # Check test MSE for every fold.
    assert np.allclose(model.cv["Test MSE"], -skl_cv["test_score"])
