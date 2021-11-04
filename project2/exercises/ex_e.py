"""
FYS-STK4155 @UiO, PROJECT II. 
Exercise e): Logistic Regression
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SKL
from sklearn.datasets import load_breast_cancer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.logistic_regression import LogisticRegression
from src.neural_network import FFNN
from src.utils import acc, scale
from src.utils import LEGEND_SIZE, LABEL_SIZE


np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
TEST_SIZE = .25


# DATA. Breast cancer.
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
X_train, X_test, _ = scale(X_train, X_test)

include_intercept = lambda X: np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
X_train = include_intercept(X_train)
X_test = include_intercept(X_test)

data = X_train, X_test, y_train, y_test


# FIT. Logistic Regression vs. Scikit-Learn.
n_epochs = 100
batch_size = 16
learning_rate = 1e-2
reg_param = 5e-3

logreg = LogisticRegression().fit(data, n_epochs, batch_size, learning_rate, reg_param)
skl = SKL(fit_intercept=False).fit(X_train, y_train)

logreg_acc = acc(logreg.predict(X_test), y_test)
skl_acc = acc(skl.predict(X_test), y_test)
print("Log. Regression:", acc(logreg.predict(X_test), y_test), np.linalg.norm(logreg.beta))
print("Scikit-Learn:", acc(skl.predict(X_test), y_test), np.linalg.norm(skl.coef_))

if SHOW_PLOTS:
    plt.plot(logreg.hist["Train"], label="train")
    plt.plot(logreg.hist["Test"], label="test")
    plt.hlines(skl_acc, 0, n_epochs, colors="k", linestyles="dashed", label="Scikit-Learn")
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("accuracy", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    if SAVE_FIGS:
        if not os.path.exists("project2/figs/"):
            os.makedirs("project2/figs/")
        plt.savefig("project2/figs/cancer_skl_vs_logreg.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# FFNN vs. Logistic Regression.
struct = [X_train.shape[1], 50, 2]
n_epochs = 50
batch_size = 8
learning_rate = 5e-2
reg_param = 5e-3

ffnn = FFNN(struct, reg_param).fit(data, n_epochs, batch_size, learning_rate)
logreg = LogisticRegression().fit(data, n_epochs, batch_size, learning_rate, reg_param)

ffnn_acc = acc(ffnn.predict(X_test), y_test)
logreg_acc = acc(logreg.predict(X_test), y_test)
print("FFNN:", acc(ffnn.predict(X_test), y_test))
print("Log. Reg.:", acc(logreg.predict(X_test), y_test))

if SHOW_PLOTS:
    clrs = iter(["C0", "C1"])

    for name, model in {"FFNN": ffnn, "Log. Regression": logreg}.items():
        clr = next(clrs)
        plt.plot(model.hist["Train"], color=clr, label=name)
        plt.plot(model.hist["Test"], color=clr, linestyle="--")

    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("accuracy", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    if SAVE_FIGS:
        plt.savefig("project2/figs/cancer_ffnn_vs_logreg.pdf", bbox_inches='tight', format="pdf")
    plt.show()
