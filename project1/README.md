# Project 1 - Regression analysis and resampling methods

## Abstract

We use ordinary least squares, Ridge, and Lasso Regression to fit Franke's function [^1] and terrain data from Møsvatn Austfjell, Norway. More specifically, we use polynomial regression models up to degree p=20. The results show that a relatively low polynomial degree of p=5 is sufficient to fit Franke's function. On the other hand, we find that for the highly irregular terrain data, high polynomial degrees of at least p=10 and a penalty method are needed to obtain good results. If a regularisation parameter is then chosen in a refined way, the terrain can be represented well, at least qualitatively.

[^1]: https://www.sfu.ca/~ssurjano/franke2d.html


## Code structure

The main functions are in the `src` folder, including
- `linear_regression.py` - An abstract base class `LinearRegression` provides the foundations for the three slightly more specific models `OLS`, `Ridge`, and `Lasso`.
- `franke.py` - Evaluation, sampling and plotting the so-called Franke's function.
- `utils.py` - Polynomial features, bias-variance analysis, bootstraping, and cross-validation.

The actual solutions to the exercises can be found in the `exercises` directory, where the functions from `src` are imported. 
The order `aux_scripts` contains helper scripts for creating a few additional figures for the report.
The files in `exercises` and `aux_scripts` both produce figures that are placed in a new local `figs` folder.


## Reproduction
To play around with the programmes yourself, please clone the repository

```
git clone https://github.com/stipesal/FYS-STK4155
```
and navigate to the first project 
```
cd FYS-STK4155/project1
```
Create a virtual environment via `python3 -m venv .venv`, activate it `source .venv/bin/activate`, and install the requirements using
```
python3 -m pip install -r requirements.txt
```
If all tests pass using `pytest -v`, you are good to go!
