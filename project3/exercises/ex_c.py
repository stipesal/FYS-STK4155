"""
FYS-STK4155 @UiO, PROJECT III. 
Exercise c): Neural PDE net.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.finite_differences import FivePoint
from src.neural_pde import HeatNet
from src.utils import LABEL_SIZE, LEGEND_SIZE


np.random.seed(2021)
torch.manual_seed(2021)

SHOW_PLOTS = True


# DATA. Diffusion coefficient. Initial datum.
c = 1.
u0 = lambda x: torch.sin(np.pi * torch.Tensor(x))
space_interval = (0, 1)
T = .5


# FDM. Implicit Euler + 2. order central
J, N = 1000 - 2, 1000 - 1
space = np.linspace(*space_interval, J + 2)
time = np.linspace(0, T, N + 1)

fdm = FivePoint((0, [0, 1, -2, 1, 0]))
fdm.solve(c, u0(space), space, time)


# PDENet.
units = [2, 50, 50, 1]
activation=nn.Tanh
pde_net = HeatNet(units, activation).set_problem(c, u0, space_interval, T)

n_epochs = 100
batch_size = 128
lr = 1e-2

pde_net.train(n_epochs, batch_size, lr)
pde_net.predict(space, time)


# PLOT. Loss.
if SHOW_PLOTS:
    plt.plot(pde_net.hist["Train"], label="train")
    plt.plot(pde_net.hist["Test"], label="test")
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("loss", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()


# PLOT. Error.
X, T = np.meshgrid(space, time)
exact_sol = np.exp(-np.pi ** 2 * T.T) * np.sin(np.pi * X.T)
models = {
    "FDM": fdm,
    "PDE-Net": pde_net,
}
for name, model in models.items():
    model.err = (model.sol - exact_sol) ** 2
    print(f"Total error - {name}: {np.linalg.norm(model.err):.3f}")

if SHOW_PLOTS:
    ext = [time.min(), time.max(), space.min(), space.max()]
    fig, axs = plt.subplots(ncols=2)
    for i, (name, model) in enumerate(models.items()):
        im = axs[i].imshow(
            model.err, extent=ext, aspect="auto", origin="lower", cmap="coolwarm",
        )
        axs[i].set_ylabel(r"$x$", size=LABEL_SIZE)
        axs[i].set_xlabel(r"$t$", size=LABEL_SIZE)
        axs[i].set_title(name)
        cbar = plt.colorbar(im, ax=axs[i-1])
        cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout()
    plt.show()

    # 3D prediction. PDENet.
    X, T = np.meshgrid(space, time)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X.T, T.T, pde_net.sol.numpy(), alpha=.9, cmap="coolwarm")
    ax.set_xlabel(r"$x$", size=LABEL_SIZE)
    ax.set_ylabel(r"$t$", size=LABEL_SIZE)
    ax.set_zlabel(r"$u(x,t)$", size=LABEL_SIZE, rotation=90)
    ax.view_init(elev=20., azim=40)
    ax.zaxis.set_rotate_label(False)
    plt.tight_layout()
    plt.show()

    # 2D. PDENet vs. exact at different time points.
    time_points = [.0, .1, .2, .3]
    idx = [np.argmin(np.abs(time - t)) for t in time_points]
    color=plt.cm.coolwarm(np.linspace(0, 1, len(time_points)))[::-1]
    for i, (ix, t) in enumerate(zip(idx, time_points)):
        plt.plot(space, pde_net.sol[:, ix], c=color[i], label=rf"$t={t:.1f}$")
        plt.plot(space, exact_sol[:, ix], "k--", alpha=.8)
    plt.plot([], [], "k--", label="exact")
    plt.xlabel(r"$x$", size=LABEL_SIZE)
    plt.ylabel(r"$u(x, t)$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.show()