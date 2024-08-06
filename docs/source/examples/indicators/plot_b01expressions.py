"""

Examples of mathematical expressions
====================================

Example of manipulating mathematical expressions and calculation of
derivatives.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 21:06:21 2023

"""

import numpy as np

from biogeme.function_output import BiogemeFunctionOutput, NamedFunctionOutput

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
# parameter 'prepare_ids' is set to True
z = expression.get_value_c(prepare_ids=True)
print(f'exp(-b * b + 1) = {z}')

# %%
# We can also calculate the value, the first derivative, the second
# derivative, and the BHHH, which in this case is the square of the
# first derivatives
the_function_output: BiogemeFunctionOutput = expression.get_value_and_derivatives(
    prepare_ids=True
)
# %%
print(f'f = {the_function_output.function}')
# %%
print(f'g = {the_function_output.gradient}')
# %%
print(f'h = {the_function_output.hessian}')
# %%
print(f'BHHH = {the_function_output.bhhh}')

# %%
# From the expression, we can create a Python function that takes as
# argument the value of the free parameters, and returns the function,
# the first, the second derivatives, and the BHHH.
fct = expression.create_function()

# %%
# We can use the function for different values of the parameter
beta = 2.0
the_named_function_output: NamedFunctionOutput = fct(beta)
print(f'f({beta}) = {the_named_function_output.function}')
print(f'g({beta}) = {the_named_function_output.gradient}')
print(f'h({beta}) = {the_named_function_output.hessian}')

# %%
beta = 3.0
the_named_function_output = fct(beta)
print(f'f({beta}) = {the_named_function_output.function}')
print(f'g({beta}) = {the_named_function_output.gradient}')
print(f'h({beta}) = {the_named_function_output.hessian}')


# %%
if can_plot:
    # We can also use it to plot the function and its derivatives
    x = np.arange(-2, 2, 0.01)

    # The value of the function is element [0].
    f = [fct(xx).function for xx in x]

    # The gradient is element [1]. As it contains only one entry [0],
    # we convert it into float.

    g = [float(fct([xx]).gradient['b']) for xx in x]

    # The hessian is element [2]. As it contains only one entry
    # [0][0], we convert it into float.
    h = [float(fct([xx]).hessian['b']['b']) for xx in x]

    ax = plt.gca()
    ax.plot(x, f, label="f(x)")
    ax.plot(x, g, label="f'(x)")
    ax.plot(x, h, label='f"(x)')
    ax.legend()

    plt.show()
