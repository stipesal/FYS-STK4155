"""
FYS-STK4155 @UiO, PROJECT I. 
Exercise 2: Bias-Variance-Tradeoff
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.linear_regression import OLS
from src.franke import sample_franke
from src.utils import design_matrix, bias_variance_analysis
from src.utils import LEGEND_SIZE, LABEL_SIZE

np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
NOISE = .1
TEST_SIZE = .2
MAX_DEGREE = 10
N_BOOTSTRAP = 50


# DATA. Uniform. Noise. Train-Test split.
N = 400
x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# LOSS. Model complexity.
train_mse, test_mse = [], []
train_r2, test_r2 = [], []
for degree in range(1, MAX_DEGREE + 1):
    X_train = design_matrix(x_train, degree=degree)
    X_test = design_matrix(x_test, degree=degree)

    model = OLS().fit(X_train, y_train)
    model.score(X_test, y_test)

    train_mse.append(model.mse_train)
    test_mse.append(model.mse_test)
    train_r2.append(model.r2_train)
    test_r2.append(model.r2_test)

if SHOW_PLOTS:
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(train_mse, "-o", label="Train")
    axs[0].plot(test_mse, "-o", label="Test")
    axs[0].set_xlabel(r"Polynomial degree $d$", size=LABEL_SIZE)
    axs[0].set_ylabel("MSE", size=LABEL_SIZE)
    axs[0].set_title("MSE")
    axs[1].plot(train_r2, "-o", label="Train")
    axs[1].plot(test_r2, "-o", label="Test")
    axs[1].set_xlabel(r"Polynomial degree $d$", size=LABEL_SIZE)
    axs[1].set_ylabel("R2", size=LABEL_SIZE)
    axs[1].set_title("R2")
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        if not os.path.exists("figs/"):
            os.makedirs("figs/")
        plt.savefig("figs/mse_r2_franke.pdf", bbox_inches='tight', format="pdf")
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
    plt.xlabel(r"Polynomial degree $d$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    if SAVE_FIGS:
        plt.savefig("figs/bias_variance_degree.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# BIAS-VARIANCE TRADEOFF. Bootstrap.
# Fixed degree, variable sample size.
degree = 5
sample_sizes = np.logspace(2, 3, 200).astype(int)
error = np.zeros((len(sample_sizes), 3))

for i, N in enumerate(sample_sizes):
    x, y = sample_franke(N, noise=NOISE)
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
    plt.xlabel(r"Sample size $N$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    if SAVE_FIGS:
        plt.savefig("figs/bias_variance_sample_size.pdf", bbox_inches='tight', format="pdf")
    plt.show()
