import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.activations import sigmoid
from src.utils import LABEL_SIZE, LEGEND_SIZE

sigmoid, _ = sigmoid()

x = np.linspace(-5, 5, 100)
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.axhline(y=0, color="k", ls="--", lw=1.)
plt.axhline(y=1, color="k", ls="--", lw=1.)
plt.xlabel(r"$x$", size=LABEL_SIZE)
plt.ylabel(r"$\sigma(x)$", size=LABEL_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
if not os.path.exists("project2/figs/"):
    os.makedirs("project2/figs/")
plt.savefig("project2/figs/sigmoid.pdf", bbox_inches='tight', format="pdf")
plt.show()
