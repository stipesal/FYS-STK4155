"""
FYS-STK4155 @UiO, PROJECT I. 
Exercise 1: Ordinary least squares
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
from franke import franke_function, sample_franke, plot_franke
from utils import plot_3d, design_matrix

np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
NOISE = .1
TEST_SIZE = .2
DEGREE = 5


# DATA. Uniform. Noise. Train-Test split.
N = 200
x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# PLOT. Train & test data.
if SHOW_PLOTS:
    ax = plot_3d()
    plot_franke(ax)
    ax.scatter3D(*x_train.T, y_train, label="Train data")
    ax.scatter3D(*x_test.T, y_test, label="Test data")
    ax.set_title("Sampling Franke's function.")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/sampling_franke.pdf", format="pdf")
    plt.show()


# OLS. Design matrix.
X_train = design_matrix(x_train, degree=DEGREE)
X_test = design_matrix(x_test, degree=DEGREE)

model = OLS().fit(X_train, y_train, confidence=.95)


# PLOT. Prediction. 2D & 3D.
if SHOW_PLOTS:
    n = 50
    x = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, x)
    mgrid = np.column_stack((x.reshape(-1), y.reshape(-1)))
    mgrid = design_matrix(mgrid, degree=DEGREE)
    # 2D.
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(franke_function(x, y))
    axs[1].imshow((model.predict(mgrid)).reshape(n, n))
    axs[0].set_title("Franke's function.")
    axs[1].set_title("Prediction.")
    plt.tight_layout()
    plt.show()
    # 3D.
    ax = plot_3d()
    plot_franke(ax)
    ax.scatter3D(*x_test.T, model.predict(X_test), c="red")
    ax.plot_wireframe(
        x, y, (model.predict(mgrid)).reshape(n, n),
        color="r",
        label="Prediction",
        alpha=.3,
    )
    ax.set_title("Predicting Franke's function.")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/prediction_3d.pdf", format="pdf")
    plt.show()


# SCORES. MSE & R2.
model.score(X_test, y_test)
print("--- SCORES ---")
print(f"MSE (Train): {model.mse_train:.4f}")
print(f"MSE (Test): {model.mse_test:.4f}")
print(f"R2 (Train): {model.r2_train:.4f}")
print(f"R2 (Test): {model.r2_test:.4f}")


# PLOT. Confidence intervals.
if SHOW_PLOTS:
    devs = .5 * (model.CI[:, 1] - model.CI[:, 0])
    plt.errorbar(range(len(model.beta)), model.beta, yerr=devs, fmt='o')
    plt.title(r"OLS estimation $\beta$.")
    plt.xticks(np.arange(len(model.beta)))
    plt.ylabel(r"$\beta_j \pm \hat{\sigma} \sqrt{(X^{\top}X)^{-1}_{jj}}$", size=14)
    plt.xlabel(r"$j$", size=14)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("figs/confidence_intervals.pdf", format="pdf")
    plt.show()
