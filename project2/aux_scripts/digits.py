import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.neural_network import FFNN
from src.utils import scale
from src.utils import LABEL_SIZE, LEGEND_SIZE


# DATA. MNIST digits.
digits = load_digits()
X = digits.images
X = X.reshape((X.shape[0], -1))
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train, X_test, _ = scale(X_train, X_test)
data = X_train, X_test, y_train, y_test


# FFNN.
structure = [X.shape[1], 10, 10, 10, 10]
reg_param = 0.0001
n_epochs = 500
batch_size = 32
lr = 1.

model = FFNN(structure, reg_param, activation="tanh").fit(data, n_epochs, batch_size, lr)
plt.plot(model.hist["Train"], label="train")
plt.plot(model.hist["Test"], label="test")
plt.xlabel("epoch", size=LABEL_SIZE)
plt.ylabel("accuracy", size=LABEL_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
if not os.path.exists("project2/figs/"):
    os.makedirs("project2/figs/")
plt.savefig("project2/figs/mnist_digits.pdf", bbox_inches='tight', format="pdf")
plt.show()
