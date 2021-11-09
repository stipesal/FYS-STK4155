# FYS-STK4155

Project work on [FYS-STK4155](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/index-eng.html) - **Applied data analysis and machine learning** @University of Oslo.


## Progress

- [x] [Project 1](https://github.com/stipesal/FYS-STK4155/tree/master/project1) - Regression analysis and resampling methods
- [ ] [Project 2](https://github.com/stipesal/FYS-STK4155/tree/master/project2) - From linear and logistic regression to neural networks
- [ ] Project 3

## Code structure

The solutions to the exercises can be found in the `project*/` folders, whereas the main functions are located in `src/`, including

- `linear_regression.py` - An abstract base class `LinearRegression` provides the foundations for the three slightly more specific models `OLS`, `Ridge`, and `Lasso`.
- `logistic_regression.py` - Simple logistic regression model trainable with (mini-batch) stochastic gradient descent.
- `neural_network.py` - Object-oriented implementation of feedforward neural networks for both, regression and classification problems.
- `activations.py` - A few common activation functions such as *(leaky) ReLU*, *sigmoid*, and *tanh*.
- `weight_inits.py` - *Xavier* and *Kaiming* weight matrix initialization.
- `optimization.py` - (Mini-batch) stochastic gradient descent.
- `franke.py` - Evaluation, sampling, and plotting the so-called Franke's function. [^1]
- `utils.py` - Utility functions such as polynomial features, bias-variance analysis, resampling techniques, and metrics.

Project-specific `aux_scripts/` contain scripts for creating additional figures for the reports. Tests for the main functions in `src/` are implemented in `test/`.

[^1]: https://www.sfu.ca/~ssurjano/franke2d.html

## Reproduction

To play around with the programmes yourself, please clone the repository

```
git clone https://github.com/stipesal/FYS-STK4155
```
and navigate to `cd FYS-STK4155/`. Create a virtual environment via `python3 -m venv .venv`, activate it `source .venv/bin/activate`, and install the requirements using
```
python3 -m pip install -r requirements.txt
```
If all tests pass using `pytest -v`, you are good to go!