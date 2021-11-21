"""
FYS-STK4155 @UiO
Testing: Neural network.
"""
import pytest
import numpy as np

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

from src.neural_network import FFNN


np.random.seed(2021)

N_SAMPLES = 1000
N_FEATURES = 10


@pytest.fixture(scope="session")
def regression_data():
    """Regression dummy data."""
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
    )
    return train_test_split(X, y, test_size=.2)


@pytest.fixture(scope="session")
def classification_data():
    """Classification dummy data."""
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
    )
    return train_test_split(X, y, test_size=.2)


def test_init(classification_data):
    """
    Tests basic initial properties such as number of neurons in each layer,
    weights and bias shapes, as well as output shapes.
    """
    reg_param = 1.

    p = [N_FEATURES, 50, 10, 2]   # Network architecture.
    L = len(p) - 2   # Number of hidden layers.

    nn = FFNN(p, reg_param)

    # No. of hidden layers + one output layer.
    assert len(nn.layers) == L + 1

    # Test weights and biases shapes.
    for i, layer in enumerate(nn.layers):
        assert layer.weights.shape == (p[i], p[i + 1])
        assert layer.bias.shape == (p[i + 1], )
    
    # Output shape.
    X = classification_data[0]
    assert nn.predict_proba(X).shape == (X.shape[0], p[-1])


def test_train(regression_data):
    """
    Tests the training procedure of the neural network
    by checking if the loss is dropping with time.
    """
    p = [N_FEATURES, 10, 1]
    reg_param = 1e-3
    lr = 1e-3
    n_epochs = 100
    batch_size=64

    nn = FFNN(p, reg_param).fit(regression_data, n_epochs, batch_size, lr)

    # Check if loss drops after every epoch.
    train_loss_diff = np.diff(nn.hist["Train"])
    test_loss_diff = np.diff(nn.hist["Test"])
    assert np.all(train_loss_diff < 0)
    assert np.all(test_loss_diff < 0)
