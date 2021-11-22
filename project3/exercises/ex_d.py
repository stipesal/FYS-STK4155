"""
FYS-STK4155 @UiO, PROJECT III. 
Exercise d): Eigenvalues.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.linalg import eig
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.neural_pde import EigenNet
from src.utils import LABEL_SIZE, LEGEND_SIZE


np.random.seed(2021)
torch.manual_seed(2021)

SHOW_PLOTS = True


# DATA. Symmetric matrix A. Rayleigh quotient.
n = 6
Q = np.random.rand(n, n)
A = (Q + Q.T) / 2
rayleigh = lambda x: (x.T @ A @ x) / (x.T @ x)


# Eigenvalues and eigenvectors.
eigen_vals, eigen_vecs = eig(A)
idx = np.argmax(eigen_vals)
true_max_eigen_val = np.real(eigen_vals[idx])
max_eigen_vec = eigen_vecs[:, idx]


# EigenNet.
units = [1, 50, 50, A.shape[0]]
activation=nn.Tanh
eigen_net = EigenNet(units, activation).set_problem(A)

n_epochs = 100
batch_size = 1000
lr = 1e-3
eigen_net.train(n_epochs, batch_size, lr)

T = 1
pred_eigen_vec = eigen_net(torch.Tensor([T])).detach().numpy()
c = (pred_eigen_vec / max_eigen_vec).mean()
pred_max_eigen_val = rayleigh(pred_eigen_vec)

print(f"True largest eigenvalue: {true_max_eigen_val:.4f}")
print(f"Predicted largest eigenvalue: {pred_max_eigen_val:.4f}")

if SHOW_PLOTS:
    hist_ = np.stack([x.detach() for x in eigen_net.eig_vec_hist[::3]])
    plt.plot(hist_)
    plt.hlines(c * max_eigen_vec, 0, n_epochs, color="k", linestyles="dashed", lw=1., label="true eigenvector")
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel(r"$N(T)$", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE, loc="lower right")
    plt.tight_layout()
    plt.show()

if SHOW_PLOTS:
    plt.plot(eigen_net.hist["Train"], label="train")
    plt.plot(eigen_net.hist["Test"], label="test")
    plt.xlabel("epoch", size=LABEL_SIZE)
    plt.ylabel("loss", size=LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.show()
