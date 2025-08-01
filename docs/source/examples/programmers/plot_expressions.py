"""

biogeme.expressions
===================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

Michel Bierlaire
Sun Jun 29 2025, 07:12:52
"""

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.calculator.function_call import (
    CallableExpression,
    function_from_expression,
)
from biogeme.calculator.simple_formula import evaluate_simple_expression
from biogeme.calculator.single_formula import (
    calculate_single_formula_from_expression,
    get_value_and_derivatives,
)
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Beta,
    BinaryMax,
    BinaryMin,
    Derive,
    Draws,
    Elem,
    IntegrateNormal,
    LinearTermTuple,
    LinearUtility,
    LogLogit,
    MonteCarlo,
    MultipleSum,
    NormalCdf,
    Numeric,
    RandomVariable,
    Variable,
    cos,
    exp,
    list_of_all_betas_in_expression,
    list_of_fixed_betas_in_expression,
    list_of_free_betas_in_expression,
    list_of_variables_in_expression,
    log,
    sin,
)
from biogeme.function_output import FunctionOutput
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.version import get_text

# %%
# Version of Biogeme.
print(get_text())

# %%
logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Simple expressions
# ------------------
# Simple expressions can be evaluated with the function `get_value`, implemented in Python. This is available
# when no database is needed.

# %%
x = Beta('x', 2, None, None, 1)
display(x)

# %%
# The `get_value`function simply returns the value of the expression if it can be evaluated.
display(x.get_value())

# %%
y = Beta('y', 3, None, None, 1)
display(y)

# %%
display(y.get_value())

# %%
one = Numeric(1)
display(one)

# %%
display(one.get_value())

# %%
# Addition
z = x + y
display(f'{x.get_value()} + {y.get_value()} = {z.get_value()}')

# %%
# Substraction
z = x - y
display(f'{x.get_value()} - {y.get_value()} = {z.get_value()}')

# %%
# Multiplication
z = x * y
display(f'{x.get_value()} * {y.get_value()} = {z.get_value()}')


# %%
# Division
z = x / y
display(f'{x.get_value()} / {y.get_value()} = {z.get_value()}')

# %%
# Power
z = x**y
z.get_value()
display(f'{x.get_value()} ** {y.get_value()} = {z.get_value()}')

# %%
# Exponential
z = exp(x)
z.get_value()
display(f'exp({x.get_value()}) = {z.get_value()}')

# %%
# Logarithm
z = log(x)
z.get_value()
display(f'log({x.get_value()}) = {z.get_value()}')

# %%
# Sine
z = sin(x)
z.get_value()
display(f'sin({x.get_value()}) = {z.get_value()}')

# %%
# Cosine
z = cos(x)
z.get_value()
display(f'cos({x.get_value()}) = {z.get_value()}')

# %%
# Minimum
z = BinaryMin(x, y)
z.get_value()
display(f'min({x.get_value()}, {y.get_value()}) = {z.get_value()}')

# %%
# Maximum
z = BinaryMax(x, y)
z.get_value()
display(f'max({x.get_value()}, {y.get_value()}) = {z.get_value()}')


# %%
# And
# An expression is considered False if its value is 0 and True otherwise.
# The outcome of a logical operator is 0 (False) or 1 (True).
z = x & y
z.get_value()
display(f'{x.get_value()} and {y.get_value()} = {z.get_value()}')

# %%
z = x & 0
z.get_value()
display(f'{x.get_value()} and 0 = {z.get_value()}')

# %%
# Or

# %%
z = x | y
z.get_value()
display(f'{x.get_value()} or {y.get_value()} = {z.get_value()}')

# %%
z = Numeric(0) | Numeric(0)
z.get_value()

# %%
# Equal
z = x == y
z.get_value()
display(f'{x.get_value()} == {y.get_value()} = {z.get_value()}')

# %%
z = (x + 1) == y
z.get_value()
display(f'({x.get_value()} + 1) == {y.get_value()} = {z.get_value()}')

# %%
# Not equal
z = x != y
z.get_value()
display(f'{x.get_value()} != {y.get_value()} = {z.get_value()}')

# %%
z = (x + 1) != y
z.get_value()
display(f'({x.get_value()} + 1) != {y.get_value()} = {z.get_value()}')


# %%
# Lesser or equal
z = x <= y
z.get_value()
display(f'{x.get_value()} <= {y.get_value()} = {z.get_value()}')


# %%
# Greater or equal
z = x >= y
z.get_value()

# %%
# Lesser than
z = x < y
z.get_value()
display(f'{x.get_value()} < {y.get_value()} = {z.get_value()}')

# %%
# Greater than
z = x > y
z.get_value()
display(f'{x.get_value()} > {y.get_value()} = {z.get_value()}')

# %%
# Opposite
z = -x
z.get_value()
display(f'-{x.get_value()} = {z.get_value()}')


# %%
# Sum of multiples expressions
list_of_expressions = [x, y, 1 + x, 1 + y]
z = MultipleSum(list_of_expressions)
all_values = [expression.get_value() for expression in list_of_expressions]
display(f'Sum of {all_values} = {z.get_value()}')

# %%
# The result is the same as the following, but it implements the sum
# in a more efficient way.
z = x + y + 1 + x + 1 + y
z.get_value()
display(f'Sum of {all_values} = {z.get_value()}')

# %%
# Element: this expression considers a dictionary of expressions, and
# an expression for the index. The index is evaluated, and the value
# of the corresponding expression in the dictionary is returned.
my_dict = {1: exp(-1), 2: log(1.2), 3: 1234}

# %%
index = x
display(f'Value of the index: {index.get_value()}')

# %%
z = Elem(my_dict, index)
z.get_value()
display(f'Value of the expression with index {index.get_value()}: {z.get_value()}')

# %%
index = x - 1
z = Elem(my_dict, index)
z.get_value()
display(f'Value of the expression with index {index.get_value()}: {z.get_value()}')


# %%
index = x - 2
index.get_value()
display(f'Value of the index: {index.get_value()}')

# %%
# If the value returned as index does not correspond to an entry in
# the dictionary, an exception is raised.

# %%
z = Elem(my_dict, index)
try:
    z.get_value()
except BiogemeError as e:
    print(f'Exception raised: {e}')

# %%
# Complex expressions
# -------------------
# When an expression is deemed complex in Biogeme, the `get_value`
# function is not available. The `evaluate_simple_expression` must be used. It is based on Jax
# to evaluate the arithmetic expressions.

# %%
# Normal CDF: it calculates
#
# .. math:: \Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}
#     e^{-\frac{1}{2}\omega^2}d\omega.

z = NormalCdf(x)
try:
    z.get_value()
except BiogemeError as e:
    print(e)

value = evaluate_simple_expression(
    expression=z, database=None, numerically_safe=False, use_jit=True
)
display(f'NormalCdf({x.get_value()}) = {value}')

# %%
value = evaluate_simple_expression(
    expression=NormalCdf(0), database=None, numerically_safe=False, use_jit=True
)
display(f'NormalCdf(0) = {value}')

# %%
# Derivative
# It is possible to calculate the derivative with respect to a parameter or a variable.
parameter = Beta('parameter', 0, None, None, 0)
variable = Variable('variable')
z = 30 * parameter + 20 * variable

# %%
dz_dparameter = Derive(z, 'parameter')
dz_dvariable = Derive(z, 'variable')

# %%
# As the expression involves a variable, it requires a database.
simple_dataframe = pd.DataFrame.from_dict({'variable': [1]})
simple_database = Database(dataframe=simple_dataframe, name='simple')
value_parameter = evaluate_simple_expression(
    expression=dz_dparameter,
    database=simple_database,
    numerically_safe=False,
    use_jit=True,
)
display(f'dz/dparameter = {value_parameter}')
value_variable = evaluate_simple_expression(
    expression=dz_dvariable,
    database=simple_database,
    numerically_safe=False,
    use_jit=True,
)
display(f'dz/variable = {value_variable}')

# %%
# Integral: let's calculate the integral of the pdf of a normal distribution:
#
# .. math:: \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{+\infty}
#     e^{-\frac{1}{2}\omega^2}d\omega = 1.

# %%
# The expression `IntegrateNormal` multiplies its argument by the pdf or the normal distribution before
# evaluating the integral. Therefore, the expression that we need to provide here is 1. However,
# Biogeme requires the expression to involve 'omega'. Therefore, we code 1 as the ratio of omega by itself.
omega = RandomVariable('omega')
z = IntegrateNormal(omega / omega, 'omega')
value = evaluate_simple_expression(
    expression=z, database=simple_database, numerically_safe=False, use_jit=True
)
display(f'The integral between -inf and +inf of the normal pdf is {value}')


# %%
# We now illustrate expressions that require a database.
# Those expressions are evaluated for each row of the database, and the obtained values are then
# added together to obtain the result.
df = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [10, 20, 30, 40, 50],
        'Variable2': [100, 200, 300, 400, 500],
        'Choice': [2, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)
my_data = Database('test', df)

# %%
# Linear utility:  it defines a linear combinations of parameters are variables.
beta1 = Beta('beta1', 10, None, None, 0)
beta2 = Beta('beta2', 20, None, None, 0)
v1 = Variable('Variable1')
v2 = Variable('Variable2')

# %%
list_of_terms = [
    LinearTermTuple(beta=beta1, x=v1),
    LinearTermTuple(beta=beta2, x=v2),
]
z = LinearUtility(list_of_terms)
value = evaluate_simple_expression(
    expression=z, database=my_data, numerically_safe=False, use_jit=True
)
display(f'beta1 * v1 + beta2 * v2 = {value}')

# %%
# It is equivalent to the following, but implemented in a more efficient way.
z = beta1 * v1 + beta2 * v2
value = evaluate_simple_expression(
    expression=z, database=my_data, numerically_safe=False, use_jit=True
)
display(f'beta1 * v1 + beta2 * v2 = {value}')

# %%
# Monte Carlo: we approximate the integral
#
# .. math:: \int_0^1 x^2 dx=\frac{1}{3}
#
# using Monte-Carlo integration. This is not considered a simple expression.
# Therefore, another function must be called.
draws = Draws('draws', 'UNIFORM')
z = MonteCarlo(draws * draws)
number_of_draws = 1_000_000
value = calculate_single_formula_from_expression(
    expression=z,
    database=simple_database,
    number_of_draws=number_of_draws,
    the_betas={},
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    numerically_safe=False,
    use_jit=True,
)
display(
    f'The Monte-Carlo approximation of the integral with {number_of_draws:_} draws is equal to {value}.'
)

# %%
# **Do not use chaining comparison expressions with Biogeme. Not only
# it does not provide the expected expression, but it does not
# trigger a warning or an exception.**
try:
    my_expression = 200 <= Variable('Variable2') <= 400
except BiogemeError as e:
    print(e)

# %%
# The correct way to code this expression is
my_expression = (200 <= Variable('Variable2')) & (Variable('Variable2') <= 400)

# %%
# The reason is that Python executes `200 <= Variable2 <= 400` as
# `(200 <= Variable2) and (Variable2 <= 400)`. The `and` operator
# cannot be overloaded in Python. Therefore, it does not return a
# Biogeme expression. Note that Pandas does not allow chaining either,
# and has implemented a `between` function instead.
my_data.dataframe['chaining_p'] = my_data.dataframe['Variable2'].between(200, 400)
display(my_data.dataframe)

# %%
# The following type of expression is another literal, corresponding
# to an unknown parameter. Note that the value is just a starting
# value for the algorithm.

beta1 = Beta('beta1', 0.2, None, None, 0)
beta2 = Beta('beta2', 0.4, None, None, 0)

# %%
# The last argument allows to fix the value of the parameter to the
# value.
beta3 = Beta('beta3', 1, None, None, 1)
beta4 = Beta('beta4', 0, None, None, 1)

# %%
# Arithmetic operators are overloaded to allow standard manipulations
# of expressions.
expr0 = beta3 + beta4
print(expr0)

# %%
# The evaluation of expressions can be done in two ways. For simple
# expressions, the function `get_value`, implemented in Python, returns
# the value of the expression.
display(f'beta3 + beta4 = {expr0.get_value()}')

# %%
# It is possible to modify the values of the parameters.
new_values = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
expr0.change_init_values(new_values)
display(f'beta3 + beta4 = {expr0.get_value()}')

# %%
# Consider another expression:
#
# .. math:: e_1 = 2 \beta_1 - \frac{\exp(-\beta_2)}{\beta_2 (\beta_3
#     \geq \beta_4) + \beta_1 (\beta_3 < \beta_4)},
#
# where :math:`(\beta_2 \geq \beta_1)` equals 1 if :math:`\beta_2 \geq
# \beta_1` and 0 otherwise.
expr1 = 2 * beta1 - exp(-beta2) / (beta2 * (beta3 >= beta4) + beta1 * (beta3 < beta4))
print(expr1)
display(f'expr1 = {expr1.get_value()}')

# %%
# It actually calls the function `get_value_and_derivatives`, and returns
# its first output (without calculating the derivatives).

value_and_derivatives: FunctionOutput = get_value_and_derivatives(
    expression=expr1,
    numerically_safe=False,
    betas={'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2},
    database=simple_database,
    gradient=True,
    hessian=True,
    bhhh=True,
    named_results=False,
    use_jit=True,
)

# %%
display(f'Value of the function: {value_and_derivatives.function}')

# %%
# We create a pandas DataFrame just to have a nicer display of the results.
# The gradient, that is the first derivatives.
display('Gradient')
display(pd.DataFrame(value_and_derivatives.gradient))

# %%
# The hessian, that is the second derivatives.
display('Hessian')
display(pd.DataFrame(value_and_derivatives.hessian))

# %%
# The BHHH matrix, that is the sum of the outer product of the gradient of each observation by itself.
display('BHHH')
display(pd.DataFrame(value_and_derivatives.bhhh))

# %%
# We illustrate the fact that the BHHH matrix is the outer product of the gradient with itself.
display('Outer product of the gradient')

display(
    pd.DataFrame(
        np.outer(
            value_and_derivatives.gradient,
            value_and_derivatives.gradient,
        )
    )
)


# %% It is possible to use obtain the results with the names of the parameters.
value_and_derivatives: FunctionOutput = get_value_and_derivatives(
    expression=expr1,
    numerically_safe=False,
    betas={'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2},
    database=simple_database,
    gradient=True,
    hessian=True,
    bhhh=True,
    named_results=True,
    use_jit=True,
)

# %%
display(f'Value of the function: {value_and_derivatives.function}')

# %%
# The gradient, that is the first derivatives.
display(value_and_derivatives.gradient)

# %%
# The hessian, that is the second derivatives.
display(pd.DataFrame(value_and_derivatives.hessian))

# %%
# The BHHH matrix, that is the sum of the outer product of the gradient of each observation by itself.
display(pd.DataFrame(value_and_derivatives.bhhh))


# %%
# It is also possible to generate a function that takes the value of the
# parameters as argument, and provides a tuple with the value of the
# expression and its derivatives. .

the_function: CallableExpression = function_from_expression(
    expression=expr1,
    database=simple_database,
    numerically_safe=False,
    use_jit=True,
    the_betas={},
)

# %%
# We evaluate it at one point...
result: FunctionOutput = the_function([1, 2], gradient=True, hessian=True, bhhh=False)
display(f'Value of the function: {result.function}')
display(f'Value of the gradient: {result.gradient}')
display(f'Value of the hessian: {result.hessian}')
display(f'The BHHH matrix has not been requested: {result.bhhh}')

# %%
# ... and at another point.
result: FunctionOutput = the_function([10, -2], gradient=True, hessian=True, bhhh=True)
display(f'Value of the function: {result.function}')
display(f'Value of the gradient: {result.gradient}')
display(f'Value of the hessian: {result.hessian}')
display(f'Value of the BHHH matrix: {result.bhhh}')

# %% If the names of the variables are needed, the parameter `named_results` must be set to True
the_function: CallableExpression = function_from_expression(
    expression=expr1,
    database=simple_database,
    numerically_safe=False,
    the_betas={},
    named_output=True,
    use_jit=True,
)

# %%
# We evaluate it at one point...
result: FunctionOutput = the_function([2, 1], gradient=True, hessian=True, bhhh=False)
display(f'Value of the function: {result.function}')
display(f'Value of the gradient: {result.gradient}')
display(f'Value of the hessian: {result.hessian}')

# %%
# The following function scans the expression and extracts a dict with
# all free parameters.
free_betas = list_of_free_betas_in_expression(the_expression=expr1)
set_of_free_beta_names = {beta.name for beta in free_betas}
display(f'free parameters: {set_of_free_beta_names}')

# %%
fixed_betas = list_of_fixed_betas_in_expression(the_expression=expr1)
set_of_fixed_beta_names = {beta.name for beta in fixed_betas}
display(f'fixed parameters: {set_of_fixed_beta_names}')

# %%
all_betas = list_of_all_betas_in_expression(the_expression=expr1)
set_of_all_beta_names = {beta.name for beta in all_betas}
display(f'all parameters: {set_of_all_beta_names}')

# %%
# The list of variables can also be extracted. In this case, there is none.
all_variables = list_of_variables_in_expression(the_expression=expr1)
set_of_variables = {variable.name for variable in all_variables}
display(f'all variables: {set_of_variables}')

# %%
# We include one variable
new_expression = expr1 + Variable('a_variable')
all_variables = list_of_variables_in_expression(the_expression=new_expression)
set_of_variables = {variable.name for variable in all_variables}
display(f'all variables: {set_of_variables}')


# %%
# We now illustrate how to calculate a logit model, that is
#
# .. math:: \frac{y_1 e^{V_1}}{y_0 e^{V_0}+y_1 e^{V_1}+y_2 e^{V_2}}
#
# where :math:`V_0=-\beta_1`, :math:`V_1=-\beta_2` and
# :math:`V_2=-\beta_1`, and :math:`y_i = 1`, :math:`i=1,2,3`.
v = {0: -beta1, 1: -beta2, 2: -beta1}
av = {0: 1, 1: 1, 2: 1}
expr7 = LogLogit(v, av, 1)
# value = evaluate_simple_expression(expression=z, database=None, numerically_safe=False)

# %%
display(f'Value of the log of the logit for alternative 1: {expr7.get_value()}')

# %%
# If the alternative is not in the choice set, an exception is raised.
expr7_wrong = LogLogit(v, av, 3)
try:
    expr7_wrong.get_value()
except BiogemeError as e:
    print(f'Exception: {e}')

# %%
# We show how to calculate the logsum.
for util in v.values():
    print(util.get_value())

# %%
logsum = np.log(
    np.sum(
        [np.exp(util.get_value()) for util in v.values()],
    )
)
display(f'Value of the logsum: {logsum}')
