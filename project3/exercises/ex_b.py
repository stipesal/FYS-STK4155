"""
FYS-STK4155 @UiO, PROJECT III. 
Exercise b): Finite differences.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.finite_differences import FivePoint
from src.utils import LABEL_SIZE, LEGEND_SIZE


SHOW_PLOTS = True


# DATA. Diffusion coefficient. Initial datum.
c = 1.

J, N = 1000 - 2, 1000 - 1
space = np.linspace(0, 1, J + 2)
time = np.linspace(0, .5, N + 1)

u0 = lambda x: np.sin(np.pi * x)


# SOLVER.
EE = FivePoint((1, [0, 1, -2, 1, 0]))
IE = FivePoint((0, [0, 1, -2, 1, 0]))
CN = FivePoint((0.5, [0, 1, -2, 1, 0]))
models = {
    "Explicit Euler": EE,
    "Implicit Euler": IE,
    "Crank-Nicolson": CN,
}
for model in models.values():
    model.solve(c, u0(space), space, time)


# PLOT. Different time points.
if SHOW_PLOTS:
    time_points = [.0, .1, .2, .3]
    idx = [np.argmin(np.abs(time - t)) for t in time_points]

    color=plt.cm.coolwarm(np.linspace(0, 1, len(time_points)))[::-1]
    for i, (ix, t) in enumerate(zip(idx, time_points)):
        plt.plot(space, CN.sol[:, ix], c=color[i], label=rf"$t={t:.1f}$")

    plt.xlabel(r"$x$", size=LABEL_SIZE)
    plt.ylabel(r"$u(x, t)$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.show()


# PLOT. Error.
X, T = np.meshgrid(space, time)
exact_sol = np.exp(-np.pi ** 2 * T.T) * np.sin(np.pi * X.T)

for model in models.values():
    model.err = (model.sol - exact_sol) ** 2

if SHOW_PLOTS:
    ext = [time.min(), time.max(), space.min(), space.max()]
    fig, axs = plt.subplots(ncols=2)
    for i, (name, model) in enumerate(models.items()):
        if name == "Explicit Euler":
            continue
        im = axs[i-1].imshow(
            model.err, extent=ext, aspect="auto", origin="lower", cmap="coolwarm",
        )
        axs[i-1].set_ylabel(r"$x$", size=LABEL_SIZE)
        axs[i-1].set_xlabel(r"$t$", size=LABEL_SIZE)
        axs[i-1].set_title(name)
        cbar = plt.colorbar(im, ax=axs[i-1])
        cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout()
    plt.show()
