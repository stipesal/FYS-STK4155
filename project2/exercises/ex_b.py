"""
FYS-STK4155 @UiO, PROJECT II. 
Exercise b): Feedforward neural network
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.linear_regression import OLS, Ridge
from src.neural_network import FFNN
from src.franke import sample_franke, plot_franke
from src.utils import plot_3d, design_matrix, mse, r2, scale
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

x_train, x_test, scaler = scale(x_train, x_test)
X_train, X_test, _ = scale(X_train, X_test)

X_train[:, 0] = 1.
X_test[:, 0] = 1.

data_lm = X_train, X_test, y_train, y_test
data = x_train, x_test, y_train, y_test


# FEATURES. 2D vs. Polynomial.
network_structure = [2, 10, 1]
n_epochs = 500
batch_size = 32
lr = 1e-3
reg_param = 1e-3

net = FFNN([2, 50, 1], reg_param).fit(data, n_epochs, batch_size, lr)
net_poly = FFNN([X_train.shape[1], 50, 1], reg_param).fit(data_lm, n_epochs, batch_size, lr)

if SHOW_PLOTS:
    plt.plot(net.hist["Train"], "C0", label="Linear")
    plt.plot(net.hist["Test"], "C0--")
    plt.plot(net_poly.hist["Train"], "C1", label="Polynomial")
    plt.plot(net_poly.hist["Test"], "C1--")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    if SAVE_FIGS:
        if not os.path.exists("project2/figs/"):
            os.makedirs("project2/figs/")
        plt.savefig("project2/figs/linear_vs_poly_features.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# GRID SEARCH. FFNN. Learning rate and regularization.
network_structure = [2, 50, 1]
n_epochs = 50
batch_size = 64

learning_rates = np.linspace(1e-4, 1e-2, 20)
reg_params = np.linspace(1e-5, 1e-1, 20)

test_mse = np.zeros((learning_rates.size, reg_params.size))
for i, lr in tqdm(enumerate(learning_rates), desc="Grid search"):
    for j, reg_param in enumerate(reg_params):
        test_ = 0.
        for _ in range(SIM):
            model = FFNN(network_structure, reg_param).fit(data, n_epochs, batch_size, lr, verbose=False)
            test_ += np.mean(model.hist["Test"][-5:])
        test_mse[i, j] = test_ / SIM

if SHOW_PLOTS:
    ext = [reg_params.min(), reg_params.max(), learning_rates.min(), learning_rates.max()]
    plt.imshow(test_mse, extent=ext, aspect="auto", cmap="coolwarm", origin="lower")
    plt.colorbar()
    plt.xlabel(r"regularization $\lambda$")
    plt.ylabel(r"learning rate")
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/gs_nn_lr_reg.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# OLS vs. Ridge vs. FFNN.
network_structure = [2, 50, 50, 1]
reg_param = 1e-4
learning_rate = 0.01
n_epochs = 1000
batch_size = 8

ols = OLS().fit(X_train, y_train)
ols_sgd = OLS().fit_sgd(data_lm, n_epochs, batch_size, learning_rate)
ridge = Ridge(reg_param).fit(X_train, y_train)
ridge_sgd = Ridge(reg_param).fit_sgd(data_lm, n_epochs, batch_size, learning_rate)
nn = FFNN(network_structure, reg_param, activation="leaky_relu").fit(data, n_epochs, batch_size, learning_rate)

models = {
    "OLS": ols,
    "OLS w/ SGD": ols_sgd,
    "Ridge": ridge,
    "Ridge w/ SGD": ridge_sgd, 
    "NN": nn,
}
for name, model in models.items():
    y_pred = model.predict(x_test) if name == "NN" else model.predict(X_test)
    mse_, r2_ = mse(y_pred, y_test), r2(y_pred, y_test)
    print(f"{name:<13}-> MSE: {mse_:.3f}, R2: {r2_:.3f}")

if SHOW_PLOTS:
    colors = iter(["C0", "C1"])
    colors_sgd = iter(["C0", "C1", "C2"])

    # MSE history.
    for name, model in models.items():
        if name in ["OLS", "Ridge"]:
            clr = next(colors)
            mse_ = mse(model.predict(X_test), y_test)
            plt.hlines(mse_, 0, n_epochs - 1, colors=clr, linestyles="dotted", label=name)
        else:
            clr = next(colors_sgd)
            plt.semilogy(model.hist["Train"], color=clr, label=name)
            plt.semilogy(model.hist["Test"], "--", color=clr)

    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("MSE", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/ols_vs_ridge_ffnn.pdf", bbox_inches='tight', format="pdf")
    plt.show()

    # Actual vs. predicted.
    colors_sgd = iter(["C0", "C1", "C2"])
    min_, max_ = np.inf, -np.inf
    for name, model in models.items():
        if name in ["OLS", "Ridge"]: continue
        y_pred = model.predict(x_test) if name == "NN" else model.predict(X_test)
        min_ = min(y_pred.min(), min_)
        max_ = max(y_pred.max(), max_)
        plt.scatter(y_test, y_pred, label=name, color=next(colors_sgd), s=10, alpha=1.)

    plt.plot([y_test.min(), y_test.max()], [min_, max_], "k--")
    plt.xlabel("Actual", size=LABEL_SIZE)
    plt.ylabel("Prediction", size=LABEL_SIZE)
    plt.axis("square")
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/actual_vs_predicted.pdf", bbox_inches='tight', format="pdf")
    plt.show()

    # 3D prediction. Neural net.
    n = 50
    x = np.linspace(0, 1, n)
    x = scaler.transform(np.column_stack((x, x)))
    x, y = np.meshgrid(*x.T)
    mgrid = np.column_stack((x.reshape(-1), y.reshape(-1)))
    ax = plot_3d()
    plot_franke(ax)
    ax.scatter3D(*scaler.inverse_transform(x_test).T, nn.predict(x_test), c="red")
    x, y = scaler.inverse_transform(np.column_stack((x[0], y[:, 0]))).T
    x, y = np.meshgrid(x, y)
    ax.plot_wireframe(
        x, y, (nn.predict(mgrid)).reshape(n, n),
        color="r",
        label="Prediction",
        alpha=.3,
    )
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/3d_prediction_nn.pdf", bbox_inches='tight', format="pdf")
    plt.show()
