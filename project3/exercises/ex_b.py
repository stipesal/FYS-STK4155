"""
FYS-STK4155 @UiO, PROJECT III. 
Exercise b): Finite differences.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import RegularGridInterpolator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.finite_differences import FivePoint, ThreePoint
from src.utils import LABEL_SIZE


SHOW_PLOTS = True
SAVE_FIGS = True


def plot_sol(models, cmap):
    ext = [time.min(), time.max(), space.min(), space.max()]
    _, axs = plt.subplots(ncols=len(models), figsize=(12, 4))
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


################
# HEAT EQUATION
################
print("---- HEAT EQUATION ----")
# DATA. Diffusion coefficient. Initial datum.
c = 1.
T = 0.5
J = 100
N = np.ceil(2 * c * T * (J + 1) ** 2).astype(int)
space = np.linspace(0, 1, J)
time = np.linspace(0, T, N)

u0 = lambda x: np.sin(np.pi * x)


# SOLVER.
FTCS = FivePoint((1, [0, 1, -2, 1, 0]))
BTCS = FivePoint((0, [0, 1, -2, 1, 0]))
CTCS = FivePoint((0.5, [0, 1, -2, 1, 0]))
models = {
    "FTCS": FTCS,
    "BTCS": BTCS,
    "CTCS": CTCS,
}
for model in models.values():
    model.solve(c, u0(space), space, time)


# PLOT. Error.
exact_sol_ = lambda X, T: np.exp(-c * np.pi ** 2 * T.T) * np.sin(np.pi * X.T)
X, T_ = np.meshgrid(space, time)
exact_sol = exact_sol_(X, T_)

for name, model in models.items():
    model.err = np.abs(model.sol - exact_sol)
    print(f"Total error - {name}: {(model.err ** 2).mean():.3e}")

if SHOW_PLOTS:
    plot_sol(models, cmap="coolwarm")
    if SAVE_FIGS:
        if not os.path.exists("project3/figs/"):
            os.makedirs("project3/figs/")
        plt.savefig("project3/figs/heat_err_comparison.pdf", bbox_inches='tight', format="pdf")
    plt.show()


#####################
# ADVECTION EQUATION
#####################
print("---- ADVECTION EQUATION ----")
# DATA. Diffusion coefficient. Initial datum.
c = 2.
T = 0.5
J = 500
N = 2 * np.ceil(c * T * (J + 1)).astype(int)
space = np.linspace(0, 1, J)
time = np.linspace(0, T, N)

u0 = lambda x: np.where((0.3 < x) & (x < 0.6), 1, 0)


# SOLVER.
FTBS = ThreePoint((1, [-1, 1, 0]))
BTBS = ThreePoint((0, [-1, 1, 0]))
CTBS = ThreePoint((0.5, [-1, 1, 0]))
models = {
    "FTBS": FTBS,
    "BTBS": BTBS,
    "CTBS": CTBS,
}
for model in models.values():
    model.solve(c, u0(space), space, time)


# PLOT. Error.
J = N = 5000
fine_space = np.linspace(0, 1, J)
fine_time = np.linspace(0, .5, N)
RF = ThreePoint((0, [-1, 1, 0]))
RF.solve(c, u0(fine_space), fine_space, fine_time)
fine_sol = RF.sol

interp = RegularGridInterpolator((fine_space, fine_time), fine_sol)

grid = np.array(np.meshgrid(space, time)).T.reshape(-1, 2)
exact_sol = interp(grid).reshape(space.size, time.size)

for name, model in models.items():
    model.err = np.abs(model.sol - exact_sol)
    print(f"Total error - {name}: {(model.err ** 2).mean():.3e}")

if SHOW_PLOTS:
    plot_sol(models, cmap="jet")
    if SAVE_FIGS:
        if not os.path.exists("project3/figs/"):
            os.makedirs("project3/figs/")
        plt.savefig("project3/figs/adv_err_comparison.pdf", bbox_inches='tight', format="pdf")
    plt.show()
