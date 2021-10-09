"""
FYS-STK4155 @UiO, PROJECT I.
Exercise 4: Ridge regression
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
)
from linear_regression import OLS, Ridge
from franke import sample_franke
from utils import design_matrix, bootstrap, bias_variance_analysis
from utils import LEGEND_SIZE, LABEL_SIZE

np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
NOISE = .1
TEST_SIZE = .2
MAX_DEGREE = 10
N_BOOTSTRAP = 100
LMBD = 1E-3


# DATA. Uniform. Noise. Train-Test split.
N = 500
x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# LOSS. Model complexity.
train_mse = []
test_mse = []
train_mse_ridge = []
test_mse_ridge = []
for degree in range(1, MAX_DEGREE + 1):
    X_train = design_matrix(x_train, degree=degree)
    X_test = design_matrix(x_test, degree=degree)

    ols = OLS().fit(X_train, y_train)
    ridge = Ridge(reg_param=LMBD).fit(X_train, y_train)

    bootstrap(ols, X_train, y_train, N_BOOTSTRAP)
    bootstrap(ridge, X_train, y_train, N_BOOTSTRAP)

    train_mse.append(np.mean(ols.boot["Train MSE"]))
    test_mse.append(np.mean(ols.boot["Test MSE"]))
    train_mse_ridge.append(np.mean(ridge.boot["Train MSE"]))
    test_mse_ridge.append(np.mean(ridge.boot["Test MSE"]))

if SHOW_PLOTS:
    plt.plot(train_mse, "k-o", label="OLS - Train")
    plt.plot(test_mse, "k--o", label="OLS - Test")
    plt.plot(train_mse_ridge, "r-o", label="Ridge - Train")
    plt.plot(test_mse_ridge, "r--o", label="Ridge - Test")
    plt.xlabel(r"Polynomial degree $d$", size=LABEL_SIZE)
    plt.ylabel("MSE", size=LABEL_SIZE)
    plt.title("Loss. OLS vs. Ridge.")
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/ridge_vs_ols.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# BIAS-VARIANCE TRADEOFF. Bootstraping.
# Fixed degree, fixed sample size, variable reg param.
DEGREE = 10
lambdas = np.logspace(-12, 3, 100)

x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)

X_train = design_matrix(x_train, degree=DEGREE)
X_test = design_matrix(x_test, degree=DEGREE)

error = np.zeros((lambdas.size, 3))
for i, lmbd in enumerate(lambdas):
    model = Ridge(reg_param=lmbd)
    error[i] = bias_variance_analysis(
        model, X_train, X_test, y_train, y_test, N_BOOTSTRAP
    )

ols = OLS()
e, b, v = bias_variance_analysis(
    ols, X_train, X_test, y_train, y_test, N_BOOTSTRAP
)

if SHOW_PLOTS:
    err, bias, var = error.T
    plt.hlines(
        [e, b, v], lambdas[0], lambdas[-1],
        linestyles="dashed", lw=1., colors=["k", "r", "b"], label="OLS",
    )
    plt.loglog(lambdas, err, "k", label='Error')
    plt.loglog(lambdas, bias, "r", label='Bias^2')
    plt.loglog(lambdas, var, "b", label='Variance')
    plt.title("Bias-variance tradeoff. Ridge.")
    plt.xlabel(r"Regularization $\lambda$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/bias_variance_ridge.pdf", bbox_inches='tight', format="pdf")
    plt.show()
