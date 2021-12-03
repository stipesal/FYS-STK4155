import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.finite_differences import FivePoint
from src.utils import LABEL_SIZE


def plot_3d():
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r"$x$", size=LABEL_SIZE)
    ax.set_ylabel(r"$t$", size=LABEL_SIZE)
    ax.set_zlabel(r"$u(x,t)$", size=LABEL_SIZE, rotation=90)
    ax.view_init(elev=20., azim=40)
    ax.zaxis.set_rotate_label(False)
    return ax


# DATA. Rough initial condition.
J, N = 1000, 1000
space = np.linspace(0, 1, J + 2)
time = np.linspace(0, .5, N + 1)

c = .5
u0 = lambda x: np.where((0.25 < x) & (x < 0.75), 1., 0.)


# SOLVER. CTCS. Central time, central space.
CTCS = FivePoint((0.5, [0, 1, -2, 1, 0]))
CTCS.solve(c, u0(space), space, time)


# PLOT. 3D.
X, T = np.meshgrid(space, time)
ax = plot_3d()
ax.plot_surface(X.T, T.T, CTCS.sol, alpha=.9, cmap="coolwarm")
ax.plot_wireframe(
    space.reshape(-1, 1), np.zeros(space.size).reshape(-1, 1), u0(space).reshape(-1, 1), color="k", lw=2.5,
)
plt.tight_layout()
if not os.path.exists("project3/figs/"):
    os.makedirs("project3/figs/")
plt.savefig("project3/figs/heat_exact_rough_init.pdf", bbox_inches='tight', format="pdf")
plt.show()
