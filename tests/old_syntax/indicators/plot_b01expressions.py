"""

Examples of mathematical expressions
====================================

Example of manipulating mathematical expressions and calculation of
derivatives.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 21:06:21 2023

"""
import numpy as np

try:
    import matplotlib.pyplot as plt

    can_plot = True
except ModuleNotFoundError:
    can_plot = False


from biogeme.expressions import Beta, exp

# ##
# We create a simple expression:
b = Beta('b', 1, None, None, 0)
expression = exp(-b * b + 1)

# %%
# We can calculate its value. Note that, as the expression is
# calculated out of Biogeme, the IDs must be prepared. So the
# parameter 'prepareIds' is set to True
z = expression.getValue_c(prepareIds=True)
print(f'exp(-b * b + 1) = {z}')

# %%
# We can also calculate the value, the first derivative, the second
# derivative, and the BHHH, which in this case is the square of the
# first derivatives
f, g, h, bhhh = expression.getValueAndDerivatives(prepareIds=True)
# %%
print(f'f = {f}')
# %%
print(f'g = {g}')
# %%
print(f'h = {h}')
# %%
print(f'BHHH = {bhhh}')

# %%
# From the expression, we can create a Python function that takes as
# argument the value of the free parameters, and returns the function,
# the first and second derivatives.
fct = expression.createFunction()

# %%
# We can use the function for different values of the parameter
beta = 2
f, g, h = fct(beta)
print(f'f({beta}) = {f}')
print(f'g({beta}) = {g}')
print(f'h({beta}) = {h}')

# %%
beta = 3
f, g, h = fct(beta)
print(f'f({beta}) = {f}')
print(f'g({beta}) = {g}')
print(f'h({beta}) = {h}')

# %%
if can_plot:
    # We can also use it to plot the function and its derivatives
    x = np.arange(-2, 2, 0.01)

    # The value of the function is element [0].
    f = [fct(xx)[0] for xx in x]

    # The gradient is element [1]. As it contains only one entry [0],
    # we convert it into float.
    g = [float(fct([xx])[1][0]) for xx in x]

    # The hessian is element [2]. As it contains only one entry
    # [0][0], we convert it into float.
    h = [float(fct([xx])[2][0][0]) for xx in x]

    ax = plt.gca()
    ax.plot(x, f, label="f(x)")
    ax.plot(x, g, label="f'(x)")
    ax.plot(x, h, label='f"(x)')
    ax.legend()

    plt.show()
