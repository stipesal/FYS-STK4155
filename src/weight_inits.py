"""
FYS-STK4155 @UiO
Weight initializations.
"""
import numpy as np


def xavier(n_input, n_output):
    """
    Returns weights and biases according to `Xavier`
    initialization using the given layer shapes.
    """
    c = np.sqrt(6. / (n_input + n_output))
    weights = np.random.uniform(-c, c, size=(n_input, n_output))
    bias = np.random.uniform(-c, c, n_output)
    return weights, bias

def kaiming(n_input, n_output):
    """
    Returns weights and biases according to `Kaiming`
    initialization using the given layer shapes.
    """
    c = np.sqrt(2 / n_input)
    weights = c * np.random.randn(n_input, n_output)
    bias = np.zeros(n_output)
    return weights, bias
