"""
FYS-STK4155 @UiO
Finite differences.
"""
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from typing import List, Optional, Tuple
from src.utils import LABEL_SIZE


class FDM:
    def __init__(self, params: Tuple[float, List[float]], name: Optional[str]=None):
        self.imex, self.coeffs = params
        if name is not None:
            self.name = name

    def set_grids(self, space: np.ndarray, time: np.ndarray):
        self.space = space
        self.time = time

        self.J = space.shape[0] - 2    # (J + 2) points in space
        self.N = time.shape[0] - 1    # (N + 1) points in time

        self.dx = space[1] - space[0]
        self.dt = time[1] - time[0]
    
    def plot_solution(self, ax=None):
        ext = [self.time.min(), self.time.max(), self.space.min(), self.space.max()]
        if ax is None:
            plt.imshow(self.sol, extent=ext, aspect="auto", origin="lower", cmap="coolwarm")
            plt.xlabel(r"$t$", size=LABEL_SIZE)
            plt.ylabel(r"$x$", size=LABEL_SIZE)
            plt.colorbar()
            plt.show()
        else:
            im = ax.imshow(self.sol, extent=ext, aspect="auto", origin="lower", cmap="coolwarm")
            ax.set_xlabel(r"$t$", size=LABEL_SIZE)
            ax.set_ylabel(r"$x$", size=LABEL_SIZE)
            return im


class FivePoint(FDM):
    def solve(self, c: float, u0: np.ndarray, space: np.ndarray, time: np.ndarray):
        self.set_grids(space, time)
        J, N = self.J, self.N

        self.alpha = c * self.dt / self.dx ** 2

        B = diags(self.coeffs, [-2, -1, 0, 1, 2], shape=(J+2, J+2))
        L = eye(J + 2) + self.alpha * (self.imex - 1) * B
        M = eye(J + 2) + self.alpha * self.imex * B

        # LU decomposition of left-hand side L.
        L = splu(L.tocsc())

        self.sol = np.zeros((J+2, N+1))
        self.sol[:, 0] = u0
        for n in range(N):
            self.sol[:, n + 1] = L.solve(M.dot(self.sol[:, n]))


class ThreePoint(FDM):
    def solve(self, c: float, u0: np.ndarray, space: np.ndarray, time: np.ndarray):
        self.set_grids(space, time)
        J, N = self.J, self.N

        alpha = c * self.dt / self.dx

        B = diags(self.coeffs, [-1, 0, 1], shape=(J+2, J+2), format="lil")
        B[0, -1] = self.coeffs[0]
        B[-1, 0] = self.coeffs[2]

        L = eye(J + 2) + alpha * (1 - self.imex) * B
        M = eye(J + 2) - alpha * self.imex * B

        # LU decomposition of left-hand side L.
        L = splu(L.tocsc())

        self.sol = np.zeros((J+2, N+1))
        self.sol[:, 0] = u0
        for n in range(N):
            self.sol[:, n + 1] = L.solve(M.dot(self.sol[:, n]))
