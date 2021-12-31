# Machine learning

Project work on [FYS-STK4155](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/index-eng.html) - **Applied data analysis and machine learning** @University of Oslo.


## Progress

- [x] [Project 1](https://github.com/stipesal/FYS-STK4155/tree/master/project1) - Regression analysis and resampling methods
- [x] [Project 2](https://github.com/stipesal/FYS-STK4155/tree/master/project2) - From linear and logistic regression to neural networks
- [x] [Project 3](https://github.com/stipesal/FYS-STK4155/tree/master/project3) - Solving partial differential equations with neural networks

## Code structure

The solutions to the exercises can be found in the `project*/` folders, whereas the main functions are located in `src/`, including among others

- [Linear regression](src/linear_regression.py) - An abstract base class `LinearRegression` provides the foundations for the three slightly more specific models `OLS`, `Ridge`, and `Lasso`.
- [Logistic regression](src/logistic_regression.py) - Simple logistic regression model trainable with (mini-batch) stochastic gradient descent.
- [Neural Networks](src/neural_network.py) - Object-oriented implementation of feedforward neural networks for both, regression and classification problems.
- [Neural PDE](src/neural_pde.py) - Implementation of `PDENet` and `EigenNet` for solving partial differential equations and eigenvalue problems.
- [Finite differences](src/finite_differences.py) - Parametrized central and upwind finite difference schemes for solving the heat and linear advection equation.

Project-specific `aux_scripts/` contain scripts for creating additional figures for the reports. Tests for the main functions in `src/` are implemented in `test/`.

## Reproduction

To play around with the programmes yourself, please clone the repository

```
git clone https://github.com/stipesal/Machine-Learning
```
and navigate to `cd Machine-Learning/`. Create a virtual environment via `python3 -m venv .venv`, activate it `source .venv/bin/activate`, and install the requirements using
```
python3 -m pip install -r requirements.txt
```
If all tests pass using `pytest -v`, you are good to go!