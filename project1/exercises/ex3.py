"""
FYS-STK4155 @UiO, PROJECT I. 
Exercise 3: Cross-validation
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
)
from linear_regression import OLS
from franke import sample_franke
from utils import design_matrix, bootstrap, cross_validation
from utils import LEGEND_SIZE, LABEL_SIZE

np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
NOISE = .1
TEST_SIZE = .2
MAX_DEGREE = 8
N_BOOTSTRAP = 100
K_FOLD = 5


# DATA. Uniform. Noise. Train-Test split.
N = 400
x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# BOOTSTRAP VS. CROSS-VALIDATION.
boot = np.zeros((MAX_DEGREE, 2))
cv = np.zeros((MAX_DEGREE, 2))

model = OLS()
for i, deg in enumerate(range(1, MAX_DEGREE + 1)):
    X_train = design_matrix(x_train, degree=deg)
    X_test = design_matrix(x_test, degree=deg)

    bootstrap(model, X_train, y_train, N_BOOTSTRAP)
    cross_validation(model, X_train, y_train, K_FOLD)

    boot[i] = np.mean(model.boot["Train MSE"]), np.mean(model.boot["Test MSE"])
    cv[i] = np.mean(model.cv["Train MSE"]), np.mean(model.cv["Test MSE"])


if SHOW_PLOTS:
    plt.plot(boot[:, 0], "r--o", label="BS - Train")
    plt.plot(boot[:, 1], "r-o", label="BS - Test")
    plt.plot(cv[:, 0], "b--o", label="CV - Train")
    plt.plot(cv[:, 1], "b-o", label="CV - Test")
    plt.title("Bootstrap vs. Cross-validation.")
    plt.xlabel(r"Polynomial degree $d$", size=LABEL_SIZE)
    plt.ylabel("MSE", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/bs_vs_cv.pdf", bbox_inches='tight', format="pdf")
    plt.show()
