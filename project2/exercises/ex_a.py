"""
FYS-STK4155 @UiO, PROJECT II. 
Exercise a): Stochastic gradient descent
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.linear_regression import OLS, Ridge
from src.franke import sample_franke
from src.utils import design_matrix, mse, r2, scale
from src.utils import LEGEND_SIZE, LABEL_SIZE


np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
NOISE = .1
TEST_SIZE = .25
DEGREE = 5
SIM = 10


# DATA. Uniform. Noise. Train-Test split.
N = 200
x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# OLS. Design matrix.
X_train = design_matrix(x_train, degree=DEGREE)
X_test = design_matrix(x_test, degree=DEGREE)

X_train, X_test, _ = scale(X_train, X_test)

X_train[:, 0] = 1.
X_test[:, 0] = 1.

data = X_train, X_test, y_train, y_test


# FITTING. Explicit vs. SGD. OLS, Ridge, and Scikit-Learn.
reg_param = 1e-3
learning_rate = 0.01
n_epochs = 500
batch_size = 16

ols = OLS().fit(X_train, y_train)
ols_sgd = OLS().fit_sgd(data, n_epochs, batch_size, learning_rate)
ridge = Ridge(reg_param).fit(X_train, y_train)
ridge_sgd = Ridge(reg_param).fit_sgd(data, n_epochs, batch_size, learning_rate)
skl = SGDRegressor().fit(X_train, y_train)

models = {
    "OLS": ols,
    "OLS w/ SGD": ols_sgd,
    "Ridge": ridge,
    "Ridge w/ SGD": ridge_sgd, 
    "Scikit-Learn": skl,
}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse_, r2_ = mse(y_pred, y_test), r2(y_pred, y_test)
    print(f"{name:<13}-> MSE: {mse_:.3f}, R2: {r2_:.3f}")

if SHOW_PLOTS:
    colors = iter(["C0", "C1", "C2"])
    colors_sgd = iter(["C0", "C1"])

    for name, model in models.items():
        if name in ["OLS", "Ridge", "sklearn"]:
            clr = next(colors)
            mse_ = mse(model.predict(X_test), y_test)
            plt.hlines(mse_, 0, n_epochs - 1, colors=clr, linestyles="dotted", lw=2., label=name)
        else:
            clr = next(colors_sgd)
            plt.semilogy(model.hist["Train"], color=clr, label=name)
            plt.semilogy(model.hist["Test"], "--", color=clr)

    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("MSE", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        if not os.path.exists("project2/figs/"):
            os.makedirs("project2/figs/")
        plt.savefig("project2/figs/sgd_vs_explicit.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# GRID SEARCH. Ridge. Learning rate and regularization.
n_epochs = 50
batch_size = 128

learning_rates = np.linspace(1e-3, 1e-2, 30)
reg_params = np.linspace(1e-5, 1e-2, 30)

test_mse = np.zeros((learning_rates.size, reg_params.size))
for i, lr in tqdm(enumerate(learning_rates), desc="Grid search (Learning rate & Reg. parameter)"):
    for j, reg_param in enumerate(reg_params):
        test_ = 0.
        for _ in range(SIM):
            model = Ridge(reg_param).fit_sgd(data, n_epochs, batch_size, lr)
            test_ += np.mean(model.hist["Test"][-5:])
        test_mse[i, j] = test_ / SIM

if SHOW_PLOTS:
    ext = [reg_params.min(), reg_params.max(), learning_rates.min(), learning_rates.max()]
    plt.imshow(test_mse, extent=ext, aspect="auto", cmap="coolwarm", origin="lower")
    plt.colorbar()
    plt.xlabel(r"regularization $\lambda$", size=LABEL_SIZE)
    plt.ylabel(r"learning rate", size=LABEL_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/gs_ridge_lr_reg.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# HYPERPARAMETER. Learning rate.
learning_rates = np.logspace(-3, -2, 200)

n_epochs = 100
batch_size = 128
reg_param = 1e-2

train_mse, test_mse = [], []
for lr in tqdm(learning_rates, desc="Learning rate analysis"):
    train_, test_ = 0, 0
    for _ in range(SIM):
        model = Ridge(reg_param).fit_sgd(data, n_epochs, batch_size, lr)
        train_ += np.mean(model.hist["Train"][-5:])
        test_ += np.mean(model.hist["Test"][-5:])
    train_mse.append(train_ / SIM)
    test_mse.append(test_ / SIM)

if SHOW_PLOTS:
    plt.loglog(learning_rates, train_mse, label="train")
    plt.loglog(learning_rates, test_mse, label="test")
    plt.loglog(learning_rates, 1e-2 * learning_rates ** -0.75, "k--", label="order 3/4")
    plt.title("RIDGE: MSE vs. learning rate")
    plt.xlabel("learning rate")
    plt.ylabel("MSE")
    plt.legend()
    if SAVE_FIGS:
        plt.savefig("project2/figs/hyper_lr.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# GRID SEARCH. Ridge. Number of epochs and batch size.
lr = 1e-2
reg_param = 1e-3

epochs = np.arange(1, 100, 4)
batch_sizes = np.arange(1, 100, 4)

test_mse = np.zeros((epochs.size, batch_sizes.size))
for i, n_epochs in tqdm(enumerate(epochs), desc="Grid search (No. epochs & Batch size)"):
    for j, batch_size in enumerate(batch_sizes):
        test_ = 0
        for _ in range(SIM):
            model = Ridge(reg_param).fit_sgd(data, n_epochs, batch_size, lr)
            test_ += model.hist["Test"][-1]
        test_mse[i, j] = test_ / SIM

if SHOW_PLOTS:
    ext = [batch_sizes.min(), batch_sizes.max(), epochs.min(), epochs.max()]
    plt.imshow(test_mse, extent=ext, aspect="auto", cmap="coolwarm", origin="lower")
    plt.colorbar()
    plt.xlabel("batch size", size=LABEL_SIZE)
    plt.ylabel("number of epochs", size=LABEL_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/gs_ridge_epochs_batch_size.pdf", bbox_inches='tight', format="pdf")
    plt.show()
