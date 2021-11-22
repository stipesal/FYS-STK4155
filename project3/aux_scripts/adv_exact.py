import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.finite_differences import ThreePoint
from src.utils import LABEL_SIZE

SHOW_PLOTS = True


# DATA. Speed coefficient. Initial datum.
c = 2.

J, N = 500, 500
space = np.linspace(0, 1, J + 2)
time = np.linspace(0, .5, N + 1)

u0 = lambda x: np.where((0.3 < x) & (x < 0.6), 1, 0)


# FDM. Implicit Euler + 2. order central
fdm = ThreePoint((0, [-1, 1, 0]))
fdm.solve(c, u0(space), space, time)


# EXACT SOLUTION.
X, T = np.meshgrid(space, time)

ax = plt.axes(projection='3d')
ax.plot_surface(X.T, T.T, fdm.sol, alpha=.9, cmap="jet")
ax.set_xlabel(r"$x$", size=LABEL_SIZE)
ax.set_ylabel(r"$t$", size=LABEL_SIZE)
ax.set_zlabel(r"$u(x,t)$", size=LABEL_SIZE, rotation=90)
ax.view_init(elev=20., azim=25)
ax.zaxis.set_rotate_label(False)
plt.show()
