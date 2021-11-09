"""
FYS-STK4155 @UiO, PROJECT I.
Exercise 5: Lasso regression
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.linear_regression import OLS, Ridge, Lasso
from src.franke import sample_franke
from src.utils import design_matrix, bootstrap, bias_variance_analysis
from src.utils import LEGEND_SIZE, LABEL_SIZE

warnings.filterwarnings("ignore")
np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
NOISE = .1
TEST_SIZE = .2
MAX_DEGREE = 10
N_BOOTSTRAP = 50
LMBD = 1E-5


# DATA. Uniform. Noise. Train-Test split.
N = 500
x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# LOSS. Model complexity.
train_mse = []
test_mse = []
train_mse_lasso = []
test_mse_lasso = []
for degree in range(1, MAX_DEGREE + 1):
    X_train = design_matrix(x_train, degree=degree)
    X_test = design_matrix(x_test, degree=degree)

    ols = OLS().fit(X_train, y_train)
    lasso = Lasso(reg_param=LMBD).fit(X_train, y_train)

    bootstrap(ols, X_train, y_train, N_BOOTSTRAP)
    bootstrap(lasso, X_train, y_train, N_BOOTSTRAP)

    train_mse.append(np.mean(ols.boot["Train MSE"]))
    test_mse.append(np.mean(ols.boot["Test MSE"]))
    train_mse_lasso.append(np.mean(lasso.boot["Train MSE"]))
    test_mse_lasso.append(np.mean(lasso.boot["Test MSE"]))

if SHOW_PLOTS:
    plt.plot(train_mse, "k-o", label="OLS - Train")
    plt.plot(test_mse, "k--o", label="OLS - Test")
    plt.plot(train_mse_lasso, "r-o", label="Lasso - Train")
    plt.plot(test_mse_lasso, "r--o", label="Lasso - Test")
    plt.xlabel(r"Polynomial degree $d$", size=LABEL_SIZE)
    plt.ylabel("MSE", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        if not os.path.exists("project1/figs/"):
            os.makedirs("project1/figs/")
        plt.savefig("project1/figs/lasso_vs_ols.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# BIAS-VARIANCE TRADEOFF. Bootstraping.
# Fixed degree, fixed sample size, variable reg param.
DEGREE = 8
lambdas = np.logspace(-5, 3, 100)

x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)

X_train = design_matrix(x_train, degree=DEGREE)
X_test = design_matrix(x_test, degree=DEGREE)

error = np.zeros((lambdas.size, 3))
for i, lmbd in enumerate(lambdas):
    model = Lasso(reg_param=lmbd)
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
    plt.xlabel(r"Regularization $\lambda$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project1/figs/bias_variance_lasso.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# Regression coefficients. OLS vs. RIDGE vs. LASSO.
lambdas = np.logspace(-5, 4, 100)

X_train = design_matrix(x_train, degree=2)
ols = OLS().fit(X_train, y_train)

betas = np.zeros((2, lambdas.size, X_train.shape[1]))
for i, lmbd in enumerate(lambdas):
    ridge = Ridge(reg_param=lmbd).fit(X_train, y_train)
    lasso = Lasso(reg_param=lmbd).fit(X_train, y_train)
    betas[0][i] = ridge.beta
    betas[1][i] = lasso.beta

if SHOW_PLOTS:
    plt.semilogx(lambdas, betas[0, :, 0], "r", label="Ridge")
    plt.semilogx(lambdas, betas[0, :, 1:], "r")
    plt.semilogx(lambdas, betas[1, :, 0], "b", label="Lasso")
    plt.semilogx(lambdas, betas[1, :, 1:], "b")
    plt.hlines(ols.beta, lambdas[0], lambdas[-1], colors="k", lw=1, linestyle="dashed", label="OLS")
    plt.xlabel(r"Regularization $\lambda$", size=LABEL_SIZE)
    plt.ylabel(r"$\beta_j$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project1/figs/regression_coeffs.pdf", bbox_inches='tight', format="pdf")
    plt.show()
