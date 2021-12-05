# Solving partial differential equations with neural networks

This study focuses on using machine learning for solving partial differential equations.
More specifically, we use fully connected feedforward neural networks trained using Adam and a loss function given by the differential equation.
The loss function is constructed using a finite set of training data coupled with problem-dependent physics, initial, and boundary conditions.
The results suggest, that the neural networks outperform established numerical solvers on coarse grids, while providing a grid-free parameterization of the solution.
We confirm the results using the 1D heat and linear advection equations.
Finally, we apply the described methods to linear eigenvalue problems.