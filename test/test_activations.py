"""
FYS-STK4155 @UiO
Testing: Activation functions.
"""
import pytest
import numpy as np

from src.activations import *


np.random.seed(2021)

activations = [
    "sigmoid",
    "relu",
    "leaky_relu",
    "softmax",
    "tanh",
]


@pytest.fixture(scope="session")
def x_space():
    """100 linearly spaced points in `(-5, 5)`."""
    return np.linspace(-5, 5, 100)


@pytest.mark.parametrize("activation", activations)
def test_eval(activation, x_space):
    """
    Tests if all activation functions
    are monotonically increasing.
    """
    f, _ = globals()[activation]()

    diffs = np.diff(f(x_space))
    assert np.all(diffs >= 0)
