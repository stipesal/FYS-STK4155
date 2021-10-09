"""
FYS-STK4155 @UiO, PROJECT I.
Exercise 6: Terrain data
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
)
from linear_regression import OLS, Ridge, Lasso
from utils import design_matrix
from utils import LEGEND_SIZE, LABEL_SIZE

warnings.filterwarnings("ignore")
np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
TEST_SIZE = .2
LMBD = 1E-4
MAX_DEGREE = 20


# LOAD TERRAIN.
terrain1 = imread(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "SRTM_data_Norway_2.tif")
)
m, n = terrain1.shape


# SAMPLE TERRAIN.
N = 2000
x = np.random.randint(n, size=(N,))
y = np.random.randint(m, size=(N,))
z = terrain1[y, x]
data = np.column_stack((x, y))


# TRAIN-TEST-SPLIT.
x_train, x_test, y_train, y_test = train_test_split(data, z, test_size=TEST_SIZE)


# SHOW DATA.
if SHOW_PLOTS:
    plt.figure()
    plt.imshow(terrain1, cmap="terrain")
    plt.scatter(*x_train.T, s=1, c="k", label="train")
    plt.scatter(*x_test.T, s=1, c="r", label="test")
    plt.xlabel(r"$X$", size=LABEL_SIZE)
    plt.ylabel(r"$Y$", size=LABEL_SIZE)
    plt.title("Terrain over Norway.")
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/sampled_terrain.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# SCALE DATA. Features normal, response in [0, 1].
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

y_train = (y_train - y_train.min(0)) / (y_train.max(0) - y_train.min(0))
y_test = (y_test - y_test.min(0)) / (y_test.max(0) - y_test.min(0))


# FIND THE BEST MODEL.
ols = OLS()
ridge = Ridge(reg_param=LMBD)
lasso = Lasso(reg_param=LMBD)

models = [ols, ridge, lasso]

mse = np.zeros((len(models), MAX_DEGREE))
for deg in range(MAX_DEGREE):
    X_train = design_matrix(x_train, degree=deg)
    X_test = design_matrix(x_test, degree=deg)

    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        mse[i][deg] = model.mse_test

if SHOW_PLOTS:
    plt.semilogy(range(MAX_DEGREE), mse[0], "k", lw=2, label="OLS")
    plt.semilogy(range(MAX_DEGREE), mse[1], "r", lw=2, label="Ridge")
    plt.semilogy(range(MAX_DEGREE), mse[2], "b", lw=2, label="Lasso")
    plt.xlabel(r"Polynomial degree $d$", size=LABEL_SIZE)
    plt.ylabel("MSE", size=LABEL_SIZE)
    plt.title("Loss. OLS vs. Ridge vs. Lasso.")
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/ols_vs_ridge_vs_lasso.pdf", bbox_inches='tight', format="pdf")
    plt.show()

best_idx = np.unravel_index(np.argmin(mse), mse.shape)
best_model = models[best_idx[0]]
best_degree = best_idx[1]

X_train = design_matrix(x_train, degree=best_degree)
best_model.fit(X_train, y_train)

print("--- BEST MODEL ---")
print(f"Model: {best_model}")
print(f"Polynomial degree: {best_degree}")
print(f"MSE: {mse[best_idx]:.4f}")


# PREDICTION. Best model.
if SHOW_PLOTS:
    proportion = .2
    n_x, n_y = int(proportion * n), int(proportion * m)  # less points for plotting.
    min_, max_ = x_train.min(0), x_train.max(0)
    x = np.linspace(min_[0], max_[0], n_x)
    y = np.linspace(min_[1], max_[1], n_y)
    x, y = np.meshgrid(x, y)
    mgrid = np.column_stack((x.reshape(-1), y.reshape(-1)))
    mesh = design_matrix(mgrid, degree=best_degree)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(terrain1, cmap="terrain")
    axs[0].set_xlabel(r"$X$", size=LABEL_SIZE)
    axs[0].set_ylabel(r"$Y$", size=LABEL_SIZE)
    axs[0].set_title("Ground truth.")
    axs[1].imshow(best_model.predict(mesh).reshape((n_y, n_x)), cmap="terrain")
    axs[1].set_xlabel(r"$X$", size=LABEL_SIZE)
    axs[1].set_ylabel(r"$Y$", size=LABEL_SIZE)
    axs[1].set_title("Prediction.")
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/terrain_pred.pdf", bbox_inches='tight', format="pdf")
    plt.show()
