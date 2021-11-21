"""
FYS-STK4155 @UiO
Testing: Neural network for PDEs.
"""
import pytest
import numpy as np
import torch

from scipy.linalg import eig
from src.neural_pde import AdvectionNet, HeatNet, EigenNet
from torch import nn


np.random.seed(2021)
torch.manual_seed(2021)

ACT = nn.Tanh
N_EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-4


@pytest.fixture(scope='session')
def c():
    """Diffusion/speed coefficent for the heat/advection equation."""
    return .5


@pytest.fixture(scope="session")
def u0():
    """Initial condition."""
    return lambda x: torch.sin(np.pi * torch.Tensor(x))


@pytest.fixture(scope="session")
def space():
    """Space domain."""
    return (0, 1)


@pytest.fixture(scope="session")
def endtime():
    """Time domain ``(0, T)``."""
    return 1.


@pytest.fixture(scope="session")
def A():
    """Returns a random and symmetric 5x5 matrix."""
    Q = np.random.rand(5, 5)
    return (Q + Q.T) / 2


@pytest.fixture(scope="session")
def heat_net(c, u0, space, endtime):
    """Solves the 1D heat equation using a neural network."""
    units = [2, 20, 20, 1]
    model = HeatNet(
        units, ACT
    ).set_problem(c, u0, space, endtime).train(N_EPOCHS, BATCH_SIZE, LR)
    return model


@pytest.fixture(scope="session")
def adv_net(c, u0, space, endtime):
    """Solves the 1D advection equation using a neural network."""
    units = [2, 20, 20, 1]
    model = AdvectionNet(
        units, ACT
    ).set_problem(c, u0, space, endtime).train(N_EPOCHS, BATCH_SIZE, LR)
    return model


@pytest.fixture(scope="session")
def eigen_net(A):
    """Solves the eigenvalue problem for `A` using a neural network.."""
    units = [1, 50, 50, A.shape[0]]
    model = EigenNet(
        units, ACT
    ).set_problem(A).train(N_EPOCHS, BATCH_SIZE, LR)
    return model


@pytest.mark.parametrize(
    "model",
    [
        "heat_net",
        "adv_net",
        "eigen_net",
    ],
)
def test_train(model, request):
    """
    Tests the training procedure of the neural networks
    by checking if the loss is dropping with time.
    """
    model = request.getfixturevalue(model)
    train_loss_diff = np.diff(model.hist["Train"])
    test_loss_diff = np.diff(model.hist["Test"])
    assert np.all(train_loss_diff < 0)
    assert np.all(test_loss_diff < 0)


def test_eigen_net(A, eigen_net):
    """
    Tests the `EigenNet` by comparing the approximated
    eigenvalue to the true maximal eigenvalue of `A`.
    """
    rayleigh_quotient = lambda x: (x.T @ A @ x) / (x.T @ x)

    T = 5
    pred_eigen_vec = eigen_net(torch.Tensor([T])).detach().numpy()
    pred_max_eigen_val = rayleigh_quotient(pred_eigen_vec)

    true_max_eigen_val = np.real(eig(A)[0].max())
    assert np.isclose(pred_max_eigen_val, true_max_eigen_val, rtol=1e-3)
