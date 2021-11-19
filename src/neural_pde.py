"""
FYS-STK4155 @UiO
Neural network for solving PDEs.
"""
import numpy as np
import torch
from torch import nn

from torch.autograd import grad
from tqdm import trange

from src.utils import LABEL_SIZE


class PDENet(nn.Module):
    def __init__(self, units, activation):
        super().__init__()
        layers = []
        for n, m in zip(units[:-1], units[1:]):
            layers.extend([nn.Linear(n, m), activation()])
        self.layers = nn.Sequential(*layers[:-1])

    def set_problem(self, c, initial_condition, space, T):
        self.c = c
        self.u0 = initial_condition
        self.space = space
        self.T = T
        return self

    def forward(self, x, t):
        input = torch.hstack((x.reshape(-1, 1), t.reshape(-1, 1)))
        return self.layers(input)

    def sample_data(self):
        N = 1000
        N_TEST = int(0.1 * N)

        xl, xr = self.space
        x_train = (xl + (xr - xl) * torch.rand(N, 1)).requires_grad_(True)
        t_train = (self.T * torch.rand(N, 1)).requires_grad_(True)
        x_test = (xl + (xr - xl) * torch.rand(N_TEST, 1)).requires_grad_(True)
        t_test = (self.T * torch.rand(N_TEST, 1)).requires_grad_(True)

        return x_train, t_train, x_test, t_test

    def loss(self, x, t):
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

    def train(self, n_epochs, batch_size, lr):
        x_train, t_train, x_test, t_test = self.sample_data()

        n_batches = x_train.size(dim=0) // batch_size
        idx = np.arange(x_train.size(dim=0))

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.hist = {"Train": [], "Test": []}
        epoch_range = trange(n_epochs, desc="Learning PDE")
        for _ in epoch_range:
            np.random.shuffle(idx)
            for b in range(n_batches):
                batch = idx[b * batch_size: (b + 1) * batch_size]

                loss = self.loss(x_train[batch], t_train[batch])
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            train_loss, test_loss = self.eval(x_train, t_train, x_test, t_test)
            epoch_range.set_postfix(train=train_loss, test=test_loss)

    def eval(self, x_train, t_train, x_test, t_test):
        train_loss = self.loss(x_train, t_train).item()
        test_loss = self.loss(x_test, t_test).item()
        self.hist["Train"].append(train_loss)
        self.hist["Test"].append(test_loss)
        return train_loss, test_loss

    def plot_solution(self, ax):
        n = 100
        space = torch.linspace(*self.space, n)
        time = torch.linspace(0, self.T, n)

        space_, time_ = torch.meshgrid(space, time, indexing="ij")
        ext = [time.min(), time.max(), space.min(), space.max()]

        pred = self(space_.reshape(-1, 1), time_.reshape(-1, 1)).detach()
        im = ax.imshow(pred.reshape(n, n), extent=ext, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_xlabel(r"$t$", size=LABEL_SIZE)
        ax.set_ylabel(r"$x$", size=LABEL_SIZE)
        return im
