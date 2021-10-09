import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
)
from utils import plot_3d
from franke import franke_function

ax = plot_3d()
x = np.linspace(0, 1, 100)
x, y = np.meshgrid(x, x)
ax.plot_surface(
    x, y, franke_function(x, y),
    cmap="twilight",
)
plt.title("Franke's function.")
plt.tight_layout()
if not os.path.exists("figs/"):
    os.makedirs("figs/")
plt.savefig("figs/franke.pdf", bbox_inches='tight', format="pdf")
plt.show()
