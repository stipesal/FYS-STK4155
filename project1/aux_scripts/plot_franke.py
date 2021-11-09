import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils import LEGEND_SIZE, plot_3d
from src.franke import franke_function

ax = plot_3d()
x = np.linspace(0, 1, 100)
x, y = np.meshgrid(x, x)
surf = ax.plot_surface(
    x, y, franke_function(x, y),
    cmap="twilight",
    label="Franke's function"
)
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
if not os.path.exists("project1/figs/"):
    os.makedirs("project1/figs/")
plt.savefig("project1/figs/franke.pdf", bbox_inches='tight', format="pdf")
plt.show()
