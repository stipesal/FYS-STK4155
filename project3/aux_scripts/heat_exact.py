import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils import LABEL_SIZE, LEGEND_SIZE


def plot_3d():
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r"$x$", size=LABEL_SIZE)
    ax.set_ylabel(r"$t$", size=LABEL_SIZE)
    ax.set_zlabel(r"$u(x,t)$", size=LABEL_SIZE, rotation=90)
    ax.view_init(elev=20., azim=40)
    ax.zaxis.set_rotate_label(False)
    return ax


# DATA. Initial datum.
J, N = 100, 100
space = np.linspace(0, 1, J + 2)
time = np.linspace(0, .5, N + 1)

u0 = lambda x: np.sin(np.pi * x)


# EXACT SOLUTION.
X, T = np.meshgrid(space, time)
exact_sol_ = lambda c: np.exp(-c * np.pi ** 2 * T.T) * np.sin(np.pi * X.T)


# PLOT. 3D.
exact_sol = exact_sol_(c=0.2)
ax = plot_3d()
ax.plot_surface(X.T, T.T, exact_sol, alpha=.9, cmap="coolwarm")
ax.plot_wireframe(
    space.reshape(-1, 1), np.zeros(space.size).reshape(-1, 1), u0(space).reshape(-1, 1), color="k", lw=2.5,
)
plt.tight_layout()
if not os.path.exists("project3/figs/"):
    os.makedirs("project3/figs/")
plt.savefig("project3/figs/heat_exact_1.pdf", bbox_inches='tight', format="pdf")
plt.show()

exact_sol = exact_sol_(c=1.0)
ax = plot_3d()
ax.plot_surface(X.T, T.T, exact_sol, alpha=.9, cmap="coolwarm")
ax.plot_wireframe(
    space.reshape(-1, 1), np.zeros(space.size).reshape(-1, 1), u0(space).reshape(-1, 1), color="k", lw=2.5,
)
plt.tight_layout()
if not os.path.exists("project3/figs/"):
    os.makedirs("project3/figs/")
plt.savefig("project3/figs/heat_exact_2.pdf", bbox_inches='tight', format="pdf")
plt.show()


# PLOT. 1D.
time_points = [.0, .1, .2, .3]
idx = [np.argmin(np.abs(time - t)) for t in time_points]

color=plt.cm.coolwarm(np.linspace(0, 1, len(time_points)))[::-1]
for i, (ix, t) in enumerate(zip(idx, time_points)):
    plt.plot(space, exact_sol[:, ix], c=color[i], label=rf"$t={t:.1f}$")

plt.xlabel(r"$x$", size=LABEL_SIZE)
plt.ylabel(r"$u(x, t)$", size=LABEL_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
if not os.path.exists("project3/figs/"):
    os.makedirs("project3/figs/")
plt.savefig("project3/figs/heat_diff_times.pdf", bbox_inches='tight', format="pdf")
plt.show()
