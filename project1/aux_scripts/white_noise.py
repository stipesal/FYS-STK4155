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
from utils import design_matrix
from utils import LEGEND_SIZE, LABEL_SIZE

np.random.seed(2021)

TEST_SIZE = .2
DEGREE = 5


# DATA. Uniform. Noise. Train-Test split.
N = 200
x, y = sample_franke(N, noise=0.)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)


# Design matrix.
X_train = design_matrix(x_train, degree=DEGREE)
X_test = design_matrix(x_test, degree=DEGREE)


# NOISE. Different levels.
noise = np.logspace(-2, 1, 100)

train_mse, test_mse = [], []

for eps in noise:   
    y_train_ = y_train + eps * np.random.randn(y_train.size)
    y_test_ = y_test + eps * np.random.randn(y_test.size)

    model = OLS().fit(X_train, y_train_)
    model.score(X_test, y_test_)

    train_mse.append(model.mse_train)
    test_mse.append(model.mse_test)


# PLOT.
plt.loglog(noise, train_mse, label="Train MSE")
plt.loglog(noise, test_mse, label="Test MSE")
plt.loglog(noise, .1 * noise**2, "k--", label="order 2")
plt.title(r"Effect of white noise $\varepsilon \sim N(0, \sigma^2)$.")
plt.xlabel(r"Variance $\sigma^2$", size=LABEL_SIZE)
plt.ylabel("MSE", size=LABEL_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
plt.savefig("figs/white_noise.pdf", bbox_inches='tight', format="pdf")
plt.show()
