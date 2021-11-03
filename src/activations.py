"""
FYS-STK4155 @UiO, PROJECT II.
Activation functions.
"""
import numpy as  np


def identity():
    f = lambda x: x
    df = lambda x: 1.
    return f, df


def sigmoid():
    f = lambda x: np.exp(x) / (1 + np.exp(x))
    df = lambda x: f(x) * (1 - f(x))
    return f, df


def relu():
    f = lambda x: np.where(x > 0, x, 0)
    df = lambda x: np.where(x > 0, 1, 0)
    return f, df


def leaky_relu():
    alpha = .1
    f = lambda x: np.where(x > 0, x, alpha * x)
    df = lambda x: np.where(x > 0, 1, alpha)
    return f, df


def softmax():
    def f(x):
        x -= x.max(axis=-1, keepdims=True)
        return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    df = lambda x: 1. # Not necessary for now.
    return f, df


def tanh():
    f = lambda x: np.tanh(x)
    df = lambda x: 1 - f(x) ** 2
    return f, df
