"""
FYS-STK4155 @UiO, PROJECT II.
Feedforward neural network.
"""
import numpy as np

from tqdm import trange

from src.activations import *
from src.weight_inits import xavier, kaiming
from src.utils import mse, acc, ohe


class Layer:
    def __init__(self, n_input, n_output, activation, weight_init):
        self.shape = (n_input, n_output)
        self.weight_init = weight_init
        self.activation = activation
        self.set_weights_and_biases()
        self.set_activation()

    def set_activation(self):
        try:
            self.act, self.d_act = globals()[self.activation]()
        except:
            raise ValueError(f"Activation {self.activation} not supported.")

    def set_weights_and_biases(self):
        try:
            self.weights, self.bias = globals()[self.weight_init](*self.shape)
        except:
            raise ValueError(f"Weight initialization {self.weight_init} not supported.")

    def forward(self, input):
        self.input = input
        self.z = self.input @ self.weights + self.bias
        return self.act(self.z)

    def update(self, reg_param, learning_rate):
        # Regularization term gradient.
        self.dW += 2 * reg_param * self.weights
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db


class Hidden(Layer):
    def __init__(self, n_input, n_output, activation, weight_init):
        super().__init__(n_input, n_output, activation, weight_init)

    def backward(self, upstream_delta):
        self.delta = upstream_delta * self.d_act(self.z)
        downstream_delta = self.delta @ self.weights.T
        self.dW = np.einsum('ij,ik->jki', self.input, self.delta).mean(axis=-1)
        self.db = self.delta.mean(axis=0)
        return downstream_delta


class Output(Layer):
    def __init__(self, n_input, n_output, activation, weight_init):
        super().__init__(n_input, n_output, activation, weight_init)
        self.weights = self.weights.squeeze()

    def backward(self, y_pred, y_true):
        if self.activation == "identity":   # Regression.
            error = 2 * (y_pred - y_true)
            self.delta = np.outer(error, self.weights)
            self.dW = (error * self.input.T).mean(axis=-1)
            self.db = error.mean(axis=-1)

        elif self.activation == "softmax":  # Classification.
            res = y_pred - ohe(y_true, y_pred.shape[1])
            self.delta = (self.weights @ y_pred.T - self.weights[:, y_true]).mean(axis=-1)
            self.dW = np.einsum('ij,ik->jki', self.input, res).mean(axis=-1)
            self.db = (y_pred - res).T.mean(axis=-1)

        return self.delta


class FFNN:
    def __init__(self, p, reg_param, activation="relu", weight_init="xavier"):
        self.p = p
        self.reg_param = reg_param
        self.activation = activation
        self.weight_init = weight_init
        self.type = (
            "regression" if self.p[-1] == 1 else "classification"
        )
        self.set_layers()

    def set_layers(self):
        self.layers = []
        for i in range(len(self.p) - 2):
            self.layers.append(
                Hidden(self.p[i], self.p[i + 1], self.activation, self.weight_init),
            )
        # Output layer activation depends on problem type.
        if self.type == "regression":
            out_activation = "identity"
        else:
            out_activation = "softmax"
        self.layers.append(
            Output(self.p[-2], self.p[-1], out_activation, self.weight_init),
        )

    def predict_proba(self, input):
        res = input
        for layer in self.layers:
            res = layer.forward(res)
        return res

    def predict(self, input):
        res = self.predict_proba(input)
        return res if self.type == "regression" else res.argmax(axis=-1)

    def backprop(self, X, y):
        y_pred = self.predict_proba(X)

        grad = self.layers[-1].backward(y_pred, y)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)
            layer.update(self.reg_param, self.learning_rate)

    def fit(self, data, n_epochs, batch_size, learning_rate, verbose=True):
        self.learning_rate = learning_rate
        X_train, _, y_train, _ = data

        n_batches = X_train.shape[0] // batch_size
        idx = np.arange(X_train.shape[0])

        self.hist = {"Train": [], "Test": []}
        if verbose: t = trange(n_epochs, desc="Train")
        else: t = range(n_epochs)
        for _ in t:
            np.random.shuffle(idx)
            for b in range(n_batches):
                batch = idx[b * batch_size: (b + 1) * batch_size]
                self.backprop(X_train[batch], y_train[batch])

            self.eval(data)
            train_, test_ = self.hist["Train"][-1], self.hist["Test"][-1]
            if verbose: t.set_postfix(train=train_, test=test_)

        return self

    def score(self, X, y):
        if self.type == "regression":
            return mse(self.predict(X), y)
        else:
            return acc(self.predict(X), y)

    def eval(self, data):
        X_train, X_test, y_train, y_test = data
        self.hist["Train"].append(self.score(X_train, y_train))
        self.hist["Test"].append(self.score(X_test, y_test))
