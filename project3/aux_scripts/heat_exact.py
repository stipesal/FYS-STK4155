import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils import LABEL_SIZE

SHOW_PLOTS = True


# DATA. Diffusion coefficient. Initial datum.
c = 1.

J, N = 100, 100
space = np.linspace(0, 1, J + 2)
time = np.linspace(0, .5, N + 1)

u0 = lambda x: np.sin(np.pi * x)

# EXACT SOLUTION.
X, T = np.meshgrid(space, time)

exact_sol = np.exp(-np.pi ** 2 * T.T) * np.sin(np.pi * X.T)

ax = plt.axes(projection='3d')
ax.plot_surface(X.T, T.T, exact_sol, alpha=.9, cmap="coolwarm")
ax.set_xlabel(r"$x$", size=LABEL_SIZE)
ax.set_ylabel(r"$t$", size=LABEL_SIZE)
ax.set_zlabel(r"$u(x,t)$", size=LABEL_SIZE, rotation=90)
ax.view_init(elev=20., azim=40)
ax.zaxis.set_rotate_label(False)
plt.show()
