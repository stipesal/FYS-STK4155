"""
FYS-STK4155 @UiO, PROJECT II. 
Exercise c): Activation functions and weight initializations
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.neural_network import FFNN
from src.activations import *
from src.franke import sample_franke
from src.utils import scale
from src.utils import LEGEND_SIZE, LABEL_SIZE


np.random.seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True
NOISE = .1
TEST_SIZE = .25
DEGREE = 5


# DATA. Uniform. Noise. Train-Test split.
N = 200
x, y = sample_franke(N, noise=NOISE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)
x_train, x_test, scaler = scale(x_train, x_test)
data = x_train, x_test, y_train, y_test


# ACTIVATION.
activation_funcs = [
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
]

if SHOW_PLOTS:
    x = np.linspace(-3, 3, 100)
    plt.axhline(y=0, color="k", ls="--")
    plt.axvline(x=0, color="k", ls="--")
    for func in activation_funcs:
        f, _ = globals()[func]()
        plt.plot(x, f(x), label=func.replace("_", " ").capitalize())
    plt.xlabel(r"$x$", size=LABEL_SIZE)
    plt.ylabel(r"$\sigma(x)$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        if not os.path.exists("project2/figs/"):
            os.makedirs("project2/figs/")
        plt.savefig("project2/figs/activations.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# FITTING.
struct = [2, 50, 50, 50, 1]
n_epochs = 200
batch_size = 16
learning_rate = 0.1
reg_param = 1e-3

nets = []
for func in activation_funcs:
    nn = FFNN(struct, reg_param, activation=func).fit(data, n_epochs, batch_size, learning_rate)
    nets.append(nn)

if SHOW_PLOTS:
    colors = iter([f"C{i}" for i in range(len(nets))])
    for net in nets:
        clr = next(colors)
        plt.semilogy(net.hist["Train"], color=clr, label=net.activation)
        plt.semilogy(net.hist["Test"], "--", color=clr)
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("MSE", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/activations_fitting.pdf", bbox_inches='tight', format="pdf")
    plt.show()


# WEIGHTS.
struct = [2, 30, 30, 1]
n_epochs = 50
batch_size = 4
lr = 0.01
reg_param = 1e-5

weight_inits = [
    "xavier",
    "kaiming",
]

if SHOW_PLOTS:
    clrs = ["C0", "C1"]

    fig, axs = plt.subplots(nrows=1, ncols=2)
    for i, weight_init in enumerate(weight_inits):
        net = FFNN(struct, reg_param, weight_init=weight_init)
        axs[i].hist(net.layers[1].weights.flatten(), bins=40, alpha=.7, color=clrs[i], density=True)
        axs[i].set_xlabel("weights", size=LABEL_SIZE)
        axs[i].set_title(weight_init.capitalize(), size=16)

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/weights_hist.pdf", bbox_inches='tight', format="pdf")
    plt.show()

    for i, weight_init in enumerate(weight_inits):
        net = FFNN(struct, reg_param, weight_init=weight_init).fit(data, n_epochs, batch_size, lr)
        plt.plot(net.hist["Train"], color=clrs[i], label=weight_init.capitalize())
        plt.plot(net.hist["Test"], color=clrs[i], linestyle="dashed")

    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig("project2/figs/weights_fitting.pdf", bbox_inches='tight', format="pdf")
    plt.show()
