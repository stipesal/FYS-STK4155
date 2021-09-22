"""
FYS-STK4155 @UiO, PROJECT I. 
Exercise 2: Bias-Variance-Tradeoff
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from utils import sample_franke_function
from utils import design_matrix
from utils import bias_variance_analysis
from utils import OLS

np.random.seed(2021)

SHOW_PLOTS = True
NOISE = .1
TEST_SIZE = .2
MAX_DEGREE = 10
N_BOOTSTRAP = 50


# DATA. Uniform. Noise. Train-Test split.
N = 400
x, y = sample_franke_function(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# LOSS. Model complexity.
train_mse = []
test_mse = []
for degree in range(1, MAX_DEGREE + 1):
    X_train = design_matrix(x_train, degree=degree)
    X_test = design_matrix(x_test, degree=degree)

    model = OLS().fit(X_train, y_train)
    model.score(X_test, y_test)

    train_mse.append(model.mse_train)
    test_mse.append(model.mse_test)

if SHOW_PLOTS:
    plt.plot(train_mse, "-o", label="Train")
    plt.plot(test_mse, "-o", label="Test")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


# BIAS-VARIANCE TRADEOFF. Bootstrap.
# Fixed sample size, variable degree.
err, bias, var = np.zeros((3, MAX_DEGREE))

for i, deg in enumerate(range(1, MAX_DEGREE + 1)):
    X_train = design_matrix(x_train, degree=deg)
    X_test = design_matrix(x_test, degree=deg)

    err[i], bias[i], var[i] = bias_variance_analysis(
        model, X_train, X_test, y_train, y_test, N_BOOTSTRAP
    )

if SHOW_PLOTS:
    plt.plot(err, "k-o", label='Error')
    plt.plot(bias, "r-o", label='Bias^2')
    plt.plot(var, "b-o", label='Variance')
    plt.title("Bias-variance tradeoff")
    plt.xlabel("Polynomial degree")
    plt.legend()
    plt.show()


# BIAS-VARIANCE TRADEOFF. Bootstrap.
# Fixed degree, variable sample size.
degree = 5
sample_sizes = np.logspace(2, 3, 200).astype(int)
error = np.zeros((len(sample_sizes), 3))

for i, N in enumerate(sample_sizes):
    x, y = sample_franke_function(N, noise=NOISE)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)

    X_train = design_matrix(x_train, degree=degree)
    X_test = design_matrix(x_test, degree=degree)

    error[i] = bias_variance_analysis(
        model, X_train, X_test, y_train, y_test, N_BOOTSTRAP
    )

if SHOW_PLOTS:
    err, bias, var = error.T
    plt.loglog(sample_sizes, err, "k", label='Error')
    plt.loglog(sample_sizes, bias, "r", label='Bias^2')
    plt.loglog(sample_sizes, var, "b", label='Variance')
    plt.title("Bias-variance tradeoff.")
    plt.xlabel("Sample size.")
    plt.legend()
    plt.show()
