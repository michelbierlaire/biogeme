"""File 01expressions.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 09:02:49 2021

Example of manipulating mathematical expressions and calculation of
derivatives.
"""
import numpy as np
import matplotlib.pyplot as plt
from biogeme.expressions import Beta, exp


# We create a simple expression
b = Beta('b', 1, None, None, 0)
expression = exp(-b * b + 1)

# We can calculate its value. Note that, as the expression is
# calculated out of Biogeme, the IDs must be prepared. So the
# parameter 'prepareIds' is set to True
z = expression.getValue_c(prepareIds=True)
print(f'exp(-b * b + 1) = {z}')

# We can also calculate the value, the first derivative, the second
# derivative, and the BHHH, which in this case is the square of the
# first derivatives

f, g, h, bhhh = expression.getValueAndDerivatives(prepareIds=True)
print(f'f = {f}')
print(f'g = {g}')
print(f'h = {h}')
print(f'BHHH = {bhhh}')

# From the expression, we can create a Python function that takes as
# argument the value of the free parameters, and returns the function,
# the first and second derivatives.
fct = expression.createFunction()

# We can use the function for different values of the parameter
beta = 2
f, g, h = fct(beta)
print(f'f({beta}) = {f}')
print(f'g({beta}) = {g}')
print(f'h({beta}) = {h}')

beta = 3
f, g, h = fct(beta)
print(f'f({beta}) = {f}')
print(f'g({beta}) = {g}')
print(f'h({beta}) = {h}')


# We can also use it to plot the function and its derivatives
x = np.arange(-2, 2, 0.01)
f = [fct(xx)[0] for xx in x]
g = [float(fct([xx])[1]) for xx in x]
h = [float(fct([xx])[2]) for xx in x]

ax = plt.gca()
ax.plot(x, f, label="f(x)")
ax.plot(x, g, label="f'(x)")
ax.plot(x, h, label='f"(x)')
ax.legend()

plt.show()
