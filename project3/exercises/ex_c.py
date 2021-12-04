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

from scipy.interpolate import RegularGridInterpolator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.finite_differences import FivePoint, ThreePoint
from src.neural_pde import HeatNet, AdvectionNet
from src.utils import LABEL_SIZE, LEGEND_SIZE


np.random.seed(2021)
torch.manual_seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True


def plot_sol_1d(models, exact_sol, time_points):
    idx = [np.argmin(np.abs(time - t)) for t in time_points]
    clrs = ["b--", "r--"]
    for ix in idx:
        plt.plot(space, exact_sol[:, ix], "k--", alpha=.8)
        for i, model in enumerate(models.values()):
            plt.plot(space, model.sol[:, ix], clrs[i], lw=1.2)
    plt.plot([], [], "k--", label="exact")
    for i, (name, model) in enumerate(models.items()):
        plt.plot([], [], clrs[i], label=name)
    plt.xlabel(r"$x$", size=LABEL_SIZE)
    plt.ylabel(r"$u(x, t)$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()


def plot_sol_2d(models, cmap):
    ext = [time.min(), time.max(), space.min(), space.max()]
    _, axs = plt.subplots(ncols=len(models), figsize=(8, 4))
    for i, (name, model) in enumerate(models.items()):
        im = axs[i].imshow(
            model.sol, extent=ext, aspect="auto", origin="lower", cmap=cmap,
        )
        axs[i].set_ylabel(r"$x$", size=LABEL_SIZE)
        axs[i].set_xlabel(r"$t$", size=LABEL_SIZE)
        axs[i].set_title(name)
        cbar = plt.colorbar(im, ax=axs[i])
        cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout()


def plot_sol_3d(models):
    clrs = ["b", "r"]
    ax = plt.axes(projection='3d')
    for i, (name, model) in enumerate(models.items()):
        if isinstance(model.err, torch.Tensor):
            ax.plot_wireframe(X.T, T_.T, model.err.numpy(), alpha=.7, color=clrs[i], label=name)
        else:
            ax.plot_wireframe(X.T, T_.T, model.err, alpha=.7, color=clrs[i], label=name)
    ax.set_xlabel(r"$x$", size=LABEL_SIZE)
    ax.set_ylabel(r"$t$", size=LABEL_SIZE)
    ax.set_zlabel(r"$u(x,t)$", size=LABEL_SIZE, rotation=90)
    ax.view_init(elev=20., azim=40)
    ax.zaxis.set_rotate_label(False)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()


################
# HEAT EQUATION
################

# DATA. Diffusion coefficient. Initial datum.
c = 1.
u0 = lambda x: torch.sin(np.pi * torch.Tensor(x))
space_interval = (0, 1)
T = .5


# FDM. Implicit Euler + 2. order central (BTCS)
J = N = 100
space = np.linspace(*space_interval, J)
time = np.linspace(0, T, N)

BTCS = FivePoint((0, [0, 1, -2, 1, 0]))
BTCS.solve(c, u0(space), space, time)


# PDENet.
units = [2, 50, 50, 1]
activation=nn.Tanh
heat_net = HeatNet(units, activation).set_problem(c, u0, space_interval, T)

n_epochs = 100
batch_size = 128
lr = 1e-2

heat_net.train(n_epochs, batch_size, lr)
heat_net.predict(space, time)


# PLOT. Loss.
if SHOW_PLOTS:
    plt.semilogy(heat_net.hist["Train"], label="train")
    plt.semilogy(heat_net.hist["Test"], label="test")
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel(r"$L(\theta)$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        if not os.path.exists("project3/figs/"):
            os.makedirs("project3/figs/")
        plt.savefig("project3/figs/heat_loss.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# PLOT. Error.
X, T_ = np.meshgrid(space, time)
exact_sol = np.exp(-np.pi ** 2 * T_.T) * np.sin(np.pi * X.T)
models = {
    "BTCS": BTCS,
    "PDENet": heat_net,
}
for name, model in models.items():
    model.err = np.abs(model.sol - exact_sol)
    print(f"Total error - {name}: {(model.err ** 2).mean():.3e}")

if SHOW_PLOTS:
    # 2D.
    plot_sol_2d(models, cmap="coolwarm")
    if SAVE_FIGS:
        plt.savefig("project3/figs/heat_err_BTCS_PDENet.pdf", bbox_inches='tight', format="pdf")
    plt.show()

    # 3D.
    plot_sol_3d(models)
    if SAVE_FIGS:
        plt.savefig("project3/figs/heat_err_3d.pdf", bbox_inches='tight', format="pdf")
    plt.show()

    # 1D. PDENet vs. BTCS vs exact.
    time_points = [.0, .1, .2, .3]
    plot_sol_1d(models, exact_sol, time_points)
    if SAVE_FIGS:
        plt.savefig("project3/figs/heat_err_1d.pdf", bbox_inches='tight', format="pdf")
    plt.show()


#####################
# ADVECTION EQUATION
#####################

# DATA. Speed coefficient. Initial datum.
c = 2.
u0 = lambda x: torch.where((0.3 < torch.Tensor(x)) & (torch.Tensor(x) < 0.6), 1, 0)
space_interval = (0, 1)
T = .5


# FDM. Implicit Euler + 1. order upwind (BTBS)
J = N = 500
space = np.linspace(*space_interval, J)
time = np.linspace(0, T, N)
X, T_ = np.meshgrid(space, time)

BTBS = ThreePoint((0, [-1, 1, 0]))
BTBS.solve(c, u0(space), space, time)


# PDENet.
units = [2, 50, 50, 50, 1]
activation=nn.Tanh
adv_net = AdvectionNet(units, activation).set_problem(c, u0, space_interval, T)

n_epochs = 200
batch_size = 128
lr = 5e-3

adv_net.train(n_epochs, batch_size, lr)
adv_net.predict(space, time)


# PLOT. Loss.
if SHOW_PLOTS:
    plt.semilogy(adv_net.hist["Train"], label="train")
    plt.semilogy(adv_net.hist["Test"], label="test")
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel(r"$L(\theta)$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        if not os.path.exists("project3/figs/"):
            os.makedirs("project3/figs/")
        plt.savefig("project3/figs/adv_loss.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# PLOT. Error.
J = N = 5000
fine_space = np.linspace(0, 1, J)
fine_time = np.linspace(0, T, N)
RF = ThreePoint((0.5, [-1, 1, 0]))
RF.solve(c, u0(fine_space), fine_space, fine_time)
fine_sol = RF.sol

interp = RegularGridInterpolator((fine_space, fine_time), fine_sol)

grid = np.array(np.meshgrid(space, time)).T.reshape(-1, 2)
exact_sol = interp(grid).reshape(space.size, time.size)

models = {
    "BTBS": BTBS,
    "PDENet": adv_net,
}
for name, model in models.items():
    model.err = np.abs(model.sol - exact_sol)
    print(f"Total error - {name}: {(model.err ** 2).mean():.3e}")

if SHOW_PLOTS:
    plot_sol_2d(models, cmap="jet")
    if SAVE_FIGS:
        plt.savefig("project3/figs/adv_err_BTBS_PDENet.pdf", bbox_inches='tight', format="pdf")
    plt.show()

    # 3D.
    plot_sol_3d(models)
    if SAVE_FIGS:
        plt.savefig("project3/figs/adv_err_3d.pdf", bbox_inches='tight', format="pdf")
    plt.show()

    # 1D.
    time_points = [.1, .4]
    plot_sol_1d(models, exact_sol, time_points)
    if SAVE_FIGS:
        plt.savefig("project3/figs/adv_err_1d.pdf", bbox_inches='tight', format="pdf")
    plt.show()
