import matplotlib.pyplot as plt
import numpy as np


MAX_DEGREE = 20
degrees = np.arange(MAX_DEGREE)

plt.plot(degrees, degrees + 1, "-o", label="homogen")
plt.plot(degrees, (degrees + 1) * (degrees + 2) / 2,   "-o", label="inhomogen")
plt.xlabel(r"Polynomial degree $d$.")
plt.ylabel("Number of monomials.")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("figs/number_of_monomials.pdf", format="pdf")
plt.show()
