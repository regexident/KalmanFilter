# KalmanFilter

A description of this package.

## Jacobian Matrix

When working with non-linear models the Kalman Filter requires one to provide Jacobian Matrices for its motion and observation model.
The Jacobian can get obtained from a given model either numerically (less efficient/precise, but more convenient by coming with batteries included) or analytically (more precise, but requires extra coding work). Thankfully with a symbolical mathematics package at hand, such as Python's [Sympy](https://sympy.org) the formula to obtain a given model's jacobian can be generated effortlessly:

1. Install Sympy

	```bash
	pip3 install sympy
	```

2. Symbolically calculate jacobian:

	```python
	from sympy import Matrix, Symbol, symbols
	from sympy import sin, cos
	
	x, y = symbols('x y')
	# Model's functions:
	X = Matrix([
		x * cos(y),
		x * sin(y),
		x**2
	])
	# State variables:
	Y = Matrix([x, y])
	
	X.jacobian(Y)
	```
3. Result:

	```python
	[
		[cos(y), -x*sin(y)],
		[sin(y),  x*cos(y)],
		[   2*x,         0]
	]
	```