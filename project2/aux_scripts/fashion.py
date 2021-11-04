import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.neural_network import FFNN
from src.utils import scale
from src.utils import LABEL_SIZE, LEGEND_SIZE


np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
TEST_SIZE = .25


# DATA. MNIST fashion.
X = np.load("project2/data/fashion_X.npy")
y = np.load("project2/data/fashion_y.npy", allow_pickle=True)

N = 10_000
X, y = X[:N], y[:N]

X = X.reshape((X.shape[0], -1))
enc = LabelEncoder().fit(y)
y = enc.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train, X_test, _ = scale(X_train, X_test)
data = X_train, X_test, y_train, y_test


# FFNN.
structure = [X.shape[1], 50, 10]
n_epochs = 100
batch_size = 128
lr = 5e-2
reg_param = 1e-3

model = FFNN(structure, reg_param, activation="tanh").fit(data, n_epochs, batch_size, lr)

plt.plot(model.hist["Train"], label="train")
plt.plot(model.hist["Test"], label="test")
plt.xlabel("epoch", size=LABEL_SIZE)
plt.ylabel("accuracy", size=LABEL_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
if not os.path.exists("project2/figs/"):
    os.makedirs("project2/figs/")
plt.savefig("project2/figs/mnist_fashion.pdf", bbox_inches='tight', format="pdf")
plt.show()
