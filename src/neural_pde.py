"""
FYS-STK4155 @UiO
Neural network for solving PDEs.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from torch import nn
from torch import Tensor
from torch.autograd import grad
from tqdm import trange
from typing import Callable, List, Tuple, Optional
from src.utils import LABEL_SIZE


N = 1000
N_TEST = 100


class ANN(nn.Module):
    def __init__(self, units: List[int], activation: Callable):
        super().__init__()
        layers = []
        for n, m in zip(units[:-1], units[1:]):
            layers.extend([nn.Linear(n, m), activation()])
        self.layers = nn.Sequential(*layers[:-1])

    def train(self, n_epochs: int, batch_size: int, lr: float) -> ANN:
        train_data, test_data = self.sample_data()

        n_batches = N // batch_size
        idx = np.arange(N)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.hist = {"Train": [], "Test": []}
        epoch_range = trange(n_epochs, desc="Learning PDE")
        for _ in epoch_range:
            np.random.shuffle(idx)
            for b in range(n_batches):
                batch = idx[b * batch_size: (b + 1) * batch_size]

                batch_train_data = [x[batch] for x in train_data]

                loss = self.loss(*batch_train_data)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            train_loss, test_loss = self.eval(train_data, test_data)
            epoch_range.set_postfix(train=train_loss, test=test_loss)
        return self

    def eval(self, train_data: Tuple[Tensor], test_data: Tuple[Tensor]) -> Tuple[float, float]:
        train_loss = self.loss(*train_data).item()
        test_loss = self.loss(*test_data).item()
        self.hist["Train"].append(train_loss)
        self.hist["Test"].append(test_loss)
        return train_loss, test_loss


class PDE_Net(ANN):
    def __init__(self, units: List[int], activation: Callable):
        super().__init__(units, activation)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        input = torch.hstack((x.reshape(-1, 1), t.reshape(-1, 1)))
        return self.layers(input)

    def sample_data(self) -> Tuple[Tuple[Tensor], Tuple[Tensor]]:
        xl, xr = self.space
        x_train = (xl + (xr - xl) * torch.rand(N, 1)).requires_grad_(True)
        t_train = (self.T * torch.rand(N, 1)).requires_grad_(True)
        x_test = (xl + (xr - xl) * torch.rand(N_TEST, 1)).requires_grad_(True)
        t_test = (self.T * torch.rand(N_TEST, 1)).requires_grad_(True)
        return (x_train, t_train), (x_test, t_test)

    def predict(self, space: np.ndarray, time: np.ndarray) -> np.ndarray:
        J, N = space.size, time.size
        space, time = torch.meshgrid(torch.Tensor(space), torch.Tensor(time), indexing="ij")
        pred = self(space.reshape(-1, 1), time.reshape(-1, 1)).detach().reshape((J, N))
        self.sol = pred
        return pred

    def plot_solution(self, ax: Optional[Axes] = None) -> AxesImage:
        if ax is None:
            _, ax = plt.subplots()
        n = 100
        space = np.linspace(*self.space, n)
        time = np.linspace(0, self.T, n)

        pred = self.predict(space, time)
        ext = [time.min(), time.max(), space.min(), space.max()]

        im = ax.imshow(pred, extent=ext, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_xlabel(r"$t$", size=LABEL_SIZE)
        ax.set_ylabel(r"$x$", size=LABEL_SIZE)
        return im


class HeatNet(PDE_Net):
    def __init__(self, units: List[int], activation: Callable):
        super().__init__(units, activation)

    def set_problem(self, c: float, initial_condition: Callable, space: Tuple[float, float], T: float) -> HeatNet:
        self.c = c
        self.u0 = initial_condition
        self.space = space
        self.T = T
        return self

    def loss(self, x: Tensor, t: Tensor) -> Tensor:
        output = self(x, t)

        # Physics.
        dx = grad(output, x, torch.ones_like(output), create_graph=True)[0]
        dx2 = grad(dx, x, torch.ones_like(dx), create_graph=True)[0]
        dt = grad(output, t, torch.ones_like(output), create_graph=True)[0]
        physics_loss = torch.mean((dt - self.c * dx2) ** 2)

        # Initial condition.
        t_init = torch.zeros_like(x)
        init_loss = torch.mean((self(x, t_init) - self.u0(x)) ** 2)

        # Boundary condition.
        xl, xr = self.space
        xl_ = xl * torch.ones_like(t)
        xr_ = xr * torch.ones_like(t)
        bound_loss = torch.mean(self(xl_, t) ** 2) + torch.mean(self(xr_, t) ** 2)

        return physics_loss + init_loss + bound_loss


class AdvectionNet(PDE_Net):
    def __init__(self, units: List[int], activation: Callable):
        super().__init__(units, activation)

    def set_problem(self, c: float, initial_condition: Callable, space: Tuple[float, float], T: float) -> HeatNet:
        self.c = c
        self.u0 = initial_condition
        self.space = space
        self.T = T
        return self

    def loss(self, x: Tensor, t: Tensor) -> Tensor:
        output = self(x, t)

        # Physics.
        dx = grad(output, x, torch.ones_like(output), create_graph=True)[0]
        dt = grad(output, t, torch.ones_like(output), create_graph=True)[0]
        physics_loss = torch.mean((dt + self.c * dx) ** 2)

        # Initial condition.
        t_init = torch.zeros_like(x)
        init_loss = torch.mean((self(x, t_init) - self.u0(x)) ** 2)

        # Boundary condition.
        xl, xr = self.space
        xl_ = xl * torch.ones_like(t)
        xr_ = xr * torch.ones_like(t)
        bound_loss = torch.mean((self(xl_, t) - self(xr_, t)) ** 2)

        return physics_loss + init_loss + bound_loss


class EigenNet(ANN):
    def __init__(self, units: List[int], activation: Callable):
        super().__init__(units, activation)
        self.eig_vec_hist = []

    def set_problem(self, A: np.ndarray) -> EigenNet:
        self.A = torch.tensor(A).float()
        self.n = A.shape[0]
        return self

    def forward(self, t: Tensor) -> Tensor:
        return self.layers(t)

    def sample_data(self) -> Tuple[Tuple[Tensor], Tuple[Tensor]]:
        T = 1
        t_train = (T * torch.rand(N, 1)).requires_grad_(True)
        t_test = (T * torch.rand(N_TEST, 1)).requires_grad_(True)
        return (t_train,), (t_test,)

    def loss(self, t: Tensor) -> Tensor:
        x = self(t)

        xx = torch.sum(x ** 2, axis=1)
        xAx = torch.sum(x.T * torch.matmul(self.A, x.T), axis=0)
        xxA = torch.einsum('i,jk->ijk', xx, self.A)
        xAxI = torch.einsum('i,jk->ijk', xAx, torch.eye(self.n))
        f = torch.einsum('ijk,ik->ij', xxA - xAxI, x)

        # Physics.
        dt = grad(x, t, torch.ones_like(x), create_graph=True)[0]
        physics_loss = torch.mean((dt - f) ** 2)

        # Initial condition.
        init_loss = torch.mean((self(torch.zeros(1).reshape(-1, 1)) - torch.ones(self.n)) ** 2)

        self.eig_vec_hist.append(self(torch.Tensor([1])))

        return physics_loss + init_loss
