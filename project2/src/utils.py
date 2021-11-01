"""
FYS-STK4155 @UiO, PROJECT II.
Useful functions such as one-hot encoding.
"""
import numpy as np


def mse(y_pred, y_true):
    """Returns the mean-squared error between predictions and true values."""
    return ((y_pred - y_true) ** 2).sum() / y_true.size

def acc(y_pred, y_true):
    """Returns the accuracy between predictions and true labels."""
    return (y_pred.argmax(axis=-1) == y_true).sum() / y_true.size

def ohe(labels, n_classes):
    """Returns a one-hot encoded array for the given labels."""
    ohe = np.zeros((labels.size, n_classes))
    ohe[np.arange(labels.size), labels] = 1.
    return ohe
