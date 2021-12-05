import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.linalg import eig
from torch import nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.neural_pde import EigenNet
from src.utils import LABEL_SIZE, LEGEND_SIZE


np.random.seed(2021)
torch.manual_seed(2021)

SHOW_PLOTS = True
SAVE_FIGS = True


def get_symmetric_matrix(n):
    Q = np.random.rand(n, n)
    return (Q + Q.T) / 2

rayleigh = lambda x: (x.T @ A @ x) / (x.T @ x)


# DATA.
T = 5
units = lambda n: [1, 100, 100, 100, n]
activation = nn.Tanh
n_epochs = 50
batch_size = 256
lr = 1e-3

ns = np.logspace(1, 2, 15).astype(int)
err = np.zeros((ns.size, 2))
for i, n in tqdm(enumerate(ns)):
    A = get_symmetric_matrix(n)

    eigen_vals, eigen_vecs = eig(A)
    idx = np.argmax(eigen_vals)
    true_max_eigen_val = np.real(eigen_vals[idx])
    max_eigen_vec = eigen_vecs[:, idx]

    eigen_net = EigenNet(units(n), activation).set_problem(A)
    eigen_net.train(n_epochs, batch_size, lr)

    pred_eigen_vec = eigen_net(torch.Tensor([T])).detach().numpy()
    c = (pred_eigen_vec / max_eigen_vec).mean()
    pred_max_eigen_val = rayleigh(pred_eigen_vec)

    err[i, 0] = np.abs(pred_max_eigen_val - true_max_eigen_val)
    err[i, 1] = np.linalg.norm(pred_eigen_vec - c * max_eigen_vec)


# PLOT. 1D.
plt.loglog(ns, err[:, 0], "ko-", label=r"$|r_{A_n}(g(T)) - \lambda_1|$")
plt.loglog(ns, err[:, 1], "ko--", label=r"$\Vert g(T) - v_{\lambda_1}\Vert_2$")
plt.loglog(ns, 10 ** -4 * ns ** 3.5, "r", lw=1., label="order 3.5")
plt.loglog(ns, 10 ** -3 * ns ** .7, "r--", lw=1., label="order 0.7")
plt.xlabel(r"$n$", size=LABEL_SIZE)
plt.ylabel("error", size=LABEL_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
if not os.path.exists("project3/figs/"):
    os.makedirs("project3/figs/")
plt.savefig("project3/figs/eigen_err_n.pdf", bbox_inches='tight', format="pdf")
plt.show()
