"""
FYS-STK4155 @UiO, PROJECT I.
Franke's function [*]: Evaluation, sampling, and plotting.

[*] http://www.dtic.mil/dtic/tr/fulltext/u2/a081688.pdf
"""
import numpy as np


def franke_function(x, y):
    """Evaluates Franke's function in `x, y`."""
    term1 = 0.75 * np.exp(-(0.25 * (9*x - 2)**2) - 0.25 * ((9*y - 2)**2))
    term2 = 0.75 * np.exp(-((9*x + 1)**2) / 49.0 - 0.1 * (9*y + 1))
    term3 = 0.50 * np.exp(-(9*x - 7)**2 / 4.0 - 0.25 * ((9*y - 3)**2))
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4


def sample_franke(N, noise):
    """
    Samples and returns `N` uniform data points in the unit square `[0,1)^2`,
    as well as the corresponding evaluation in Franke's function.
    """
    X = np.random.rand(N, 2)
    x, y = X.T

    Y = franke_function(x, y)
    Y += noise * np.random.randn(N)
    return X, Y


def plot_franke(ax):
    """Plots Franke's function as a wireframe on the given axes."""
    x = np.linspace(0, 1, 50)
    x, y = np.meshgrid(x, x)
    ax.plot_wireframe(
        x, y, franke_function(x, y),
        color="k",
        label="Franke's function",
        alpha=.3,
    )
