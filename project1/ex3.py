"""
FYS-STK4155 @UiO, PROJECT I. 
Exercise 3: Cross-validation
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from utils import sample_franke_function
from utils import design_matrix
from utils import bootstrap
from utils import cross_validation
from utils import LinearRegression

np.random.seed(2021)

SHOW_PLOTS = True
NOISE = .1
TEST_SIZE = .2
MAX_DEGREE = 8
N_BOOTSTRAP = 50
K_FOLD = 5


# DATA. Uniform. Noise. Train-Test split.
N = 1000
X, Y = sample_franke_function(N, noise=NOISE)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE)


# BOOTSTRAP VS. CROSS-VALIDATION.
boot = np.zeros((MAX_DEGREE, 2))
cv = np.zeros((MAX_DEGREE, 2))

model = LinearRegression()
for i, deg in enumerate(range(1, MAX_DEGREE + 1)):
    X_train_ = design_matrix(X_train, degree=deg)
    X_test_ = design_matrix(X_test, degree=deg)

    bootstrap(model, X_train_, Y_train, N_BOOTSTRAP)
    cross_validation(model, X_train_, Y_train, K_FOLD)

    boot[i] = np.mean(model.boot["Train MSE"]), np.mean(model.boot["Test MSE"])
    cv[i] = np.mean(model.cv["Train MSE"]), np.mean(model.cv["Test MSE"])


if SHOW_PLOTS:
    plt.plot(boot[:, 0], "r--o", label="BS - Train")
    plt.plot(boot[:, 1], "r-o", label="BS - Test")
    plt.plot(cv[:, 0], "b--o", label="CV - Train")
    plt.plot(cv[:, 1], "b-o", label="CV - Test")
    plt.title("Bootstrap vs. cross-validation")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
