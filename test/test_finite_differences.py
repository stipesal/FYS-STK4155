"""
FYS-STK4155 @UiO
Testing: Finite differences.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.finite_differences import ThreePoint, FivePoint


SHOW_PLOTS = True


@pytest.fixture(scope='session')
def c():
    """Returns the diffusion coefficent in the heat equation."""
    return .3


@pytest.fixture(scope="session")
def u0():
    """Initial condition."""
    return lambda x: np.sin(np.pi * x)


@pytest.fixture(scope="session")
def space():
    return np.linspace(0, 1, 1000)


@pytest.fixture(scope="session")
def time():
    return np.linspace(0, 0.5, 1000)


@pytest.fixture(scope="session")
def solver_heat(c, u0, space, time):
    """
    Solves the heat equation with the FDM
    (Implicit Euler + 2. order central)
    and returns the solver object.
    """
    RF = FivePoint((0, np.array([0, 1, -2, 1, 0])))
    RF.solve(c, u0(space), space, time)
    if SHOW_PLOTS:
        RF.plot_solution()
    return RF


@pytest.fixture(scope='session')
def solver_adv(c, u0, space, time):
    """
    Solves the advection equation with the FDM
    (Implicit Euler + 1. order upwind)
    and returns the solver object.
    """
    RF = ThreePoint((0, np.array([-1, 1, 0])))
    RF.solve(c, u0(space), space, time)
    if SHOW_PLOTS:
        RF.plot_solution()
    return RF


def test_heat_eq_inital_condition(solver_heat, u0):
    assert np.all(solver_heat.sol[:, 0] == u0(solver_heat.space))


def test_heat_eq_boundary_condition(solver_heat):
    assert np.allclose(solver_heat.sol[[0, -1]], np.zeros((2, solver_heat.N + 1)), atol=1e-2)


def test_adv_eq_conservation(solver_adv):
    row_sum = solver_adv.sol.sum(axis=0)
    assert np.allclose(row_sum, row_sum[0] * np.ones(row_sum.size))


def test_adv_eq_initial_condition(solver_adv, u0):
    assert np.all(solver_adv.sol[:, 0] == u0(solver_adv.space))


def test_adv_eq_boundary_condition(solver_adv):
    assert np.allclose(solver_adv.sol[0], solver_adv.sol[-1], atol=1E-1)
