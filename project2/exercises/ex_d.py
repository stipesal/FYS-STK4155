"""
FYS-STK4155 @UiO, PROJECT II. 
Exercise d): Classification
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.neural_network import FFNN
from src.utils import scale, acc
from src.utils import LEGEND_SIZE, LABEL_SIZE


np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
TEST_SIZE = .25
SIM = 10


# DATA. Breast cancer.
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
X_train, X_test, _ = scale(X_train, X_test)
data = X_train, X_test, y_train, y_test


# FITTING. Scikit-Learn vs. FFNN.
structure = [X.shape[1], 20, 20, 2]
n_epochs = 100
batch_size = 32
reg_param = 1e-4
learning_rate = 5e-1

ffnn = FFNN(structure, reg_param).fit(data, n_epochs, batch_size, learning_rate)
skl = MLPClassifier().fit(X_train, y_train)

ffnn_acc = acc(ffnn.predict(X_test), y_test)
skl_acc = acc(skl.predict(X_test), y_test)
print(f"FFNN: {ffnn_acc:.7f}")
print(f"SKL: {skl_acc:.7f}")

if SHOW_PLOTS:
    plt.plot(ffnn.hist["Train"], label="train")
    plt.plot(ffnn.hist["Test"], label="test")
    plt.hlines(skl_acc, 0, n_epochs, color="k", linestyles="dashed", label="Scikit-Learn")
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("accuracy", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        if not os.path.exists("project2/figs/"):
            os.makedirs("project2/figs/")
        plt.savefig("project2/figs/cancer_skl_vs_ffnn.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# Number of neurons w/ one hidden layer.
n_epochs = 100
batch_size = 32
lr = 5e-2
reg_param = 1e-4
neurons = np.arange(2, 20, 2)

train_mse, test_mse = [], []
for n_neurons in tqdm(neurons):
    struct = [X.shape[1], n_neurons, 2]
    train_, test_ = 0., 0.
    for _ in range(SIM):
        net = FFNN(struct, reg_param).fit(data, n_epochs, batch_size, lr, verbose=False)
        train_ += np.mean(net.hist["Train"][-5:])
        test_ += np.mean(net.hist["Test"][-5:])
    train_mse.append(train_ / SIM)
    test_mse.append(test_ / SIM)

if SHOW_PLOTS:
    plt.plot(neurons, train_mse, "o-", label="train")
    plt.plot(neurons, test_mse, "o-", label="test")
    plt.xlabel("number of neurons", size=LABEL_SIZE)
    plt.ylabel("accuracy", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/acc_vs_no_of_neurons.pdf", bbox_inches='tight', format="pdf")
    plt.show()
