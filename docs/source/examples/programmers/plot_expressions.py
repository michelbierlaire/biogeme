"""

biogeme.expressions
===================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Tue Nov 21 14:53:49 2023
"""

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    IdManager,
    TypeOfElementaryExpression,
    LinearTermTuple,
    Beta,
    Numeric,
    exp,
    log,
    bioMin,
    bioMax,
    bioMultSum,
    Elem,
    bioNormalCdf,
    Derive,
    RandomVariable,
    Integrate,
    Variable,
    bioLinearUtility,
    bioDraws,
    MonteCarlo,
    PanelLikelihoodTrajectory,
    _bioLogLogit,
    LogLogit,
)
from biogeme.function_output import (
    BiogemeDisaggregateFunctionOutput,
    BiogemeFunctionOutput,
    NamedBiogemeFunctionOutput,
    NamedBiogemeDisaggregateFunctionOutput,
)
from biogeme.models import loglogit, logit
from biogeme.tools import check_derivatives
from biogeme.version import get_text

# %%
# Version of Biogeme.
print(get_text())

# %%
logger = blog.get_screen_logger(level=blog.DEBUG)

# %%
# Simple expressions
# ------------------
# Simple expressions can be evaluated both with the functions
# `get_value`(implemented in Python) and the `get_value_c` (implemented
# in C++). They do not require a database.

# %%
x = Beta('x', 2, None, None, 1)
display(x)

# %%
x.get_value()

# %%
x.get_value_c(prepare_ids=True)


# %%
y = Beta('y', 3, None, None, 1)
display(y)

# %%
y.get_value()

# %%
y.get_value_c(prepare_ids=True)

# %%
# Note that if the parameter has to be estimated, its value cannot be obtained.
unknown_parameter = Beta('x', 2, None, None, 0)
try:
    unknown_parameter.get_value()
except BiogemeError as e:

    print(e)

# %%
one = Numeric(1)
display(one)

# %%
one.get_value()

# %%
one.get_value_c(prepare_ids=True)

# %%
# Addition
z = x + y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Substraction
z = x - y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Multiplication
z = x * y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Division
z = x / y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Power
z = x**y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Exponential
z = exp(x)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Logarithm
z = log(x)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Minimum
z = bioMin(x, y)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Maximum
z = bioMax(x, y)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# And
z = x & y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
z = x & 0
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Or

# %%
z = x | y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
z = Numeric(0) | Numeric(0)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Equal
z = x == y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
z = (x + 1) == y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Not equal
z = x != y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
z = (x + 1) != y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Lesser or equal
z = x <= y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Greater or equal
z = x >= y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Lesser than
z = x < y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Greater than
z = x > y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Opposite
z = -x
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Sum of multiples expressions
listOfExpressions = [x, y, 1 + x, 1 + y]
z = bioMultSum(listOfExpressions)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# The result is the same as the following, but it implements the sum
# in a more efficient way.
z = x + y + 1 + x + 1 + y
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
# Element: this expression considers a dictionary of expressions, and
# an expression for the index. The index is evaluated, and the value
# of the corresponding expression in the dictionary is returned.
my_dict = {1: exp(-1), 2: log(1.2), 3: 1234}

# %%
index = x
index.get_value()

# %%
z = Elem(my_dict, index)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
index = x - 1
index.get_value()

# %%
z = Elem(my_dict, index)
z.get_value()

# %%
z.get_value_c(prepare_ids=True)

# %%
index = x - 2
index.get_value()

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
z = Elem(my_dict, index)
try:
    z.get_value_c(prepare_ids=True)
except RuntimeError as e:
    print(f'Exception raised: {e}')

# %%
# Complex expressions
# -------------------
# When an expression is deemed complex in Biogeme, the `get_value`
# function is not available. Only the `get_value_c` function must be
# used. It calculates the expressions using a C++ implementation of
# the expression.

# %%
# Normal CDF: it calculates
#
# .. math:: \Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}
#     e^{-\frac{1}{2}\omega^2}d\omega.
z = bioNormalCdf(x)
z.get_value_c(prepare_ids=True)

# %%
z = bioNormalCdf(0)
z.get_value_c(prepare_ids=True)

# %%
# Derivative
z = 30 * x + 20 * y

# %%
zx = Derive(z, 'x')
zx.get_value_c(prepare_ids=True)

# %%
zx = Derive(z, 'y')
zx.get_value_c(prepare_ids=True)

# %%
# Integral: let's calculate the integral of the pdf of a normal distribution:
#
# .. math:: \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{+\infty}
#     e^{-\frac{1}{2}\omega^2}d\omega = 1.
omega = RandomVariable('omega')
pdf = exp(-omega * omega / 2)
z = Integrate(pdf, 'omega') / np.sqrt(2 * np.pi)
z.get_value_c(prepare_ids=True)

# %%
# In order to change the bounds of integration, a change of variables
# must be performed. Let's calculate
#
# .. math:: \int_0^1 x^2 dx=\frac{1}{3}.
#
# If :math:`a` is the lower bound of integration, and :math:`b` is the upper
# bound, the change of variable is
#
# .. math:: x = a + \frac{b-a}{1+e^{-\omega}},
#
# and
#
# .. math:: dx = \frac{(b-a)e^{-\omega}}{(1+e^{-\omega})^2}d\omega.
a = 0
b = 1
t = a + (b - a) / (1 + exp(-omega))
dt = (b - a) * exp(-omega) * (1 + exp(-omega)) ** -2
integrand = t * t
z = Integrate(integrand * dt / (b - a), 'omega')
z.get_value_c(prepare_ids=True)

# %%
# Expressions using a database
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
z = bioLinearUtility(list_of_terms)
z.get_value_c(database=my_data, prepare_ids=True)

# %%
# It is equivalent to the following, but implemented in a more efficient way.
z = beta1 * v1 + beta2 * v2
z.get_value_c(database=my_data, prepare_ids=True)

# %%
# Monte Carlo: we approximate the integral
#
# .. math:: \int_0^1 x^2 dx=\frac{1}{3}
#
# using Monte-Carlo integration. As draws require a database, it is
# calculated for each entry in the database.
draws = bioDraws('draws', 'UNIFORM')
z = MonteCarlo(draws * draws)
z.get_value_c(database=my_data, prepare_ids=True)

# %%
# Panel Trajectory: we first calculate a quantity for each entry in the database.
v1 = Variable('Variable1')
v2 = Variable('Variable2')
p = v1 / (v1 + v2)
p.get_value_c(database=my_data, prepare_ids=True)

# %%
# We now declare the data as "panel", based on the identified
# `Person`. It means that the first three rows correspond to a
# sequence of three observations for individual 1, and the last two, a
# sequence of two observations for individual 2. The panel trajectory
# calculates the expression for each row associated with an
# individual, and calculate the product.
my_data.panel('Person')

# %%
# In this case, we expect the following for individual 1:
display(0.09090909**3)

# %%
# And the following for individual 2:
display(0.09090909**2)

# %%
# We verify that it is indeed the case:
z = PanelLikelihoodTrajectory(p)
z.get_value_c(database=my_data, prepare_ids=True)


# %%
# More complex expressions
# ------------------------
# We set the number of draws for Monte-Carlo integration. It should be
# a large number. For the sake of computational efficiency, as this
# notebook is designed to illustrate the various function, we use a
# low value.

# %%
NUMBER_OF_DRAWS = 100

# %%
# We first create a small database
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
display(df)

# %%
my_data = Database('test', df)

# %%
# The following type of expression is a literal called Variable that
# corresponds to an entry in the database.
Person = Variable('Person')
Variable1 = Variable('Variable1')
Variable2 = Variable('Variable2')
Choice = Variable('Choice')
Av1 = Variable('Av1')
Av2 = Variable('Av2')
Av3 = Variable('Av3')

# %%
# It is possible to add a new column to the database, that creates a
# new variable that can be used in expressions.
newvar_b = my_data.define_variable('newvar_b', Variable1 + Variable2)
display(my_data.data)

# %%
# It is equivalent to the following Pandas statement.
my_data.data['newvar_p'] = my_data.data['Variable1'] + my_data.data['Variable2']
display(my_data.data)

# %%
# **Do not use chaining comparison expressions with Biogeme. Not only
# it does not provide the expected expression, but it does not
# trigger a warning or an exception.**
try:
    my_expression = 200 <= Variable2 <= 400
except BiogemeError as e:
    print(e)


# %%
# The reason is that Python executes `200 <= Variable2 <= 400` as
# `(200 <= Variable2) and (Variable2 <= 400)`. The `and` operator
# cannot be overloaded in Python. Therefore, it does not return a
# Biogeme expression. Note that Pandas does not allow chaining either,
# and has implemented a `between` function instead.
my_data.data['chaining_p'] = my_data.data['Variable2'].between(200, 400)
display(my_data.data)

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
expr0.get_value()

# %%
# It is possible to modify the values of the parameters.
newvalues = {'beta1': 1, 'beta2': 2, 'beta3': 3, 'beta4': 2}
expr0.change_init_values(newvalues)
expr0.get_value()

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


# %%
# The function `get_value_c` is implemented in C++, and works for any
# expression. When use outside a specific context, the IDs must be
# explicitly prepared.
expr1.get_value_c(prepare_ids=True)

# %%
# It actually calls the function `get_value_and_derivatives`, and returns
# its first output (without calculating the derivatives).
the_output = expr1.get_value_and_derivatives(prepare_ids=True)

# %%
display(the_output.function)

# %%
# We create a pandas DataFrame just to have a nicer display of the results.
pd.DataFrame(the_output.gradient)

# %%
pd.DataFrame(the_output.hessian)

# %%
pd.DataFrame(the_output.bhhh)

# %%
# Note that the BHHH matrix is the outer product of the gradient with itself.
pd.DataFrame(np.outer(the_output.gradient, the_output.gradient))

# %%
# If the derivatives are not needed, their calculation can be
# skipped. Here, we calculate the gradient, but not the hessian.
expr1.get_value_and_derivatives(
    gradient=True, hessian=False, bhhh=False, prepare_ids=True
)

# %% It is possible to use obtain the results with the names of the parameters.
the_output: NamedBiogemeFunctionOutput = expr1.get_value_and_derivatives(
    prepare_ids=True, named_results=True
)

# %%
print(the_output.gradient)

# %%
print(the_output.hessian)

# %%
print(the_output.bhhh)

# %%
# It can also generate a function that takes the value of the
# parameters as argument, and provides a tuple with the value of the
# expression and its derivatives. By default, it returns the value of
# the function, its gradient and its hessian.
the_function = expr1.create_function()

# %%
# We evaluate it at one point...
the_function([1, 2])

# %%
# ... and at another point.
the_function([10, -2])

# %%
# We can use it to check the derivatives.
check_derivatives(the_function, np.array([1, 2]), logg=True)

# %%
# And it is possible to also obtain the BHHH matrix.
the_function = expr1.create_function(bhhh=True)
the_function([1, 2])

# %%
# It can take a database as input, and evaluate the expression and its
# derivatives for each entry in the database.  In the following
# example, as no variable of the database is involved in the
# expression, the output of the expression is the same for each entry.
results: BiogemeDisaggregateFunctionOutput = expr1.get_value_and_derivatives(
    database=my_data, aggregation=False
)
print(len(results.functions))

# %%
f_array = results.functions
g_array = results.gradients
h_array = results.hessians
bhhh_array = results.bhhhs

for f, g, h, bhhh in zip(f_array, g_array, h_array, bhhh_array):
    print('******')
    print(f'{f=}')
    print(f'{g=}')
    print(f'{h=}')
    print(f'{bhhh=}')

# %% If the names of the variables are needed, the parameter `named_results` must be set to True

results: NamedBiogemeDisaggregateFunctionOutput = expr1.get_value_and_derivatives(
    database=my_data, aggregation=False, named_results=True
)
print(len(results.functions))

# %%
f_array = results.functions
g_array = results.gradients
h_array = results.hessians
bhhh_array = results.bhhhs

for f, g, h, bhhh in zip(f_array, g_array, h_array, bhhh_array):
    print('******')
    print(f'{f=}')
    print(f'{g=}')
    print(f'{h=}')
    print(f'{bhhh=}')

# %%
# If `aggregation` is set to `True`, the results are accumulated as a sum.
the_output: BiogemeFunctionOutput = expr1.get_value_and_derivatives(
    database=my_data, aggregation=True
)
print(f'{the_output.function=}')
print(f'{the_output.gradient=}')
print(f'{the_output.hessian=}')
print(f'{the_output.bhhh=}')

print('B3')
# %%
# The following function scans the expression and extracts a dict with
# all free parameters.
expr1.set_of_elementary_expression(TypeOfElementaryExpression.FREE_BETA)

# %%
# Options can be set to extract free parameters, fixed parameters, or both.
expr1.set_of_elementary_expression(TypeOfElementaryExpression.FIXED_BETA)

# %%
expr1.set_of_elementary_expression(TypeOfElementaryExpression.BETA)

# %%
# It is possible also to extract an elementary expression from its name.
expr1.get_elementary_expression('beta2')

# %%
# Let's consider an expression involving two variables :math:`V_1` and
# :math:`V_2`:
#
# .. math:: e_2 = 2 \beta_1 V_1 - \frac{\exp(-\beta_2 V_2)}{\beta_2 (\beta_3
#     \geq \beta_4) + \beta_1 (\beta_3 < \beta_4)},
#
# where :math:`(\beta_2 \geq \beta_1)` equals 1 if :math:`\beta_2 \geq
# \beta_1` and 0 otherwise. Note that, in our example, the second term
# is numerically negligible with respect to the first one.

expr2 = 2 * beta1 * Variable1 - exp(-beta2 * Variable2) / (
    beta2 * (beta3 >= beta4) + beta1 * (beta3 < beta4)
)
print(expr2)

print('B4')
# %%
# It is not a simple expression anymore, and only the function
# `get_value_c` can be invoked. If we try the `get_value` function, it
# raises an exception.
try:
    expr2.get_value()
except BiogemeError as e:
    print(f'Exception raised: {e}')

# %%
# As the expression is called out of a specific context, it should be
# instructed to prepare its IDs. Note that if no database is
# provided, an exception is raised when the formula contains
# variables. Indeed, the values of these variables cannot be found
# anywhere.
try:
    expr2.get_value_c(prepare_ids=True)
except BiogemeError as e:
    print(f'Exception raised: {e}')

print('B5')
# %%
expr2.get_value_c(database=my_data, aggregation=False, prepare_ids=True)

# %%
# The following function extracts the names of the parameters
# appearing in the expression.
expr2.set_of_elementary_expression(TypeOfElementaryExpression.BETA)

# %%
# The list of parameters can also be obtained in the form of a dictionary.
expr2.dict_of_elementary_expression(TypeOfElementaryExpression.BETA)

# %%
# The list of variables can also be obtained in the form of a dictionary.
expr2.dict_of_elementary_expression(TypeOfElementaryExpression.VARIABLE)

# %%
# or a set...
expr2.set_of_elementary_expression(TypeOfElementaryExpression.VARIABLE)

# %%
# Expressions are defined recursively, using a tree
# representation. The following function describes the type of the
# uppermost node of the tree.
expr2.get_class_name()

# %%
# The signature is a formal representation of the expression,
# assigning identifiers to each node of the tree, and representing
# them starting from the leaves. It is easy to parse, and is passed to
# the C++ implementation.

# %%
# As the expression is used out of a specific context, it must be
# prepared before using it.
expr2.prepare(database=my_data, number_of_draws=0)
expr2.get_status_id_manager()
print(expr2)

# %%
expr2.get_signature()

# %%
# The elementary expressions are
#
# - free parameters,
# - fixed parameters,
# - random variables (for numerical integration),
# - draws (for Monte-Carlo integration), and
# - variables from the database.
#
# The following function extracts all elementary expressions from a
# list of formulas, give them a unique numbering, and return them
# organized by group, as defined above (except for the
# variables, that are directly available in the database).
collection_of_formulas = [expr1, expr2]
formulas = IdManager(collection_of_formulas, my_data, None)


# %%
# Unique numbering for all elementary expressions.
display(formulas.elementary_expressions.indices)

# %%
display(formulas.free_betas)

# %%
# Each elementary expression has two ids. One unique index across all
# elementary expressions, and one unique within each specific group.
display(
    [(i.elementaryIndex, i.betaId) for k, i in formulas.free_betas.expressions.items()]
)

# %%
display(formulas.free_betas.names)

# %%
display(formulas.fixed_betas)

# %%
display(
    [(i.elementaryIndex, i.betaId) for k, i in formulas.fixed_betas.expressions.items()]
)

# %%
display(formulas.fixed_betas.names)

# %%
display(formulas.random_variables)

# %%
# Monte Carlo integration is based on draws.
my_draws = bioDraws('my_draws', 'UNIFORM')
expr3 = MonteCarlo(my_draws * my_draws)
print(expr3)

# %%
# Note that draws are different from random variables, used for
# numerical integration.
expr3.set_of_elementary_expression(TypeOfElementaryExpression.RANDOM_VARIABLE)

# %%
# The following function reports the draws involved in an expression.
expr3.set_of_elementary_expression(TypeOfElementaryExpression.DRAWS)

# %%
# The following function checks if draws are defined outside
# MonteCarlo, and return their names.
wrong_expression = my_draws + MonteCarlo(my_draws * my_draws)
wrong_expression.check_draws()

# %%
# Checking the correct expression returns an empty set.
expr3.check_draws()

# %%
# The expression is a Monte-Carlo integration.
expr3.get_class_name()

# %%
# Note that the draws are associated with a database. Therefore, the
# evaluation of expressions involving Monte Carlo integration can only
# be done on a database. If none is provided, an exception is raised.
try:
    expr3.get_value_c(number_of_draws=NUMBER_OF_DRAWS)
except BiogemeError as e:
    print(f'Exception raised: {e}')

# %%
# Here is its value. It is an approximation of
#
# .. math:: \int_0^1 x^2 dx=\frac{1}{3}.
expr3.get_value_c(database=my_data, number_of_draws=NUMBER_OF_DRAWS, prepare_ids=True)

# %%
# Here is its signature.
expr3.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
expr3.get_signature()

# %%
# The same integral can be calculated using numerical integration,
# declaring a random variable.
omega = RandomVariable('omega')

# %%
# Numerical integration calculates integrals between :math:`-\infty` and
# :math:`+\infty`. Here, the interval being :math:`[0,1]`, a change of variables
# is required.
a = 0
b = 1
x = a + (b - a) / (1 + exp(-omega))
dx = (b - a) * exp(-omega) * (1 + exp(-omega)) ** (-2)
integrand = x * x
expr4 = Integrate(integrand * dx / (b - a), 'omega')

# %%
# In this case, omega is a random variable.
expr4.dict_of_elementary_expression(TypeOfElementaryExpression.RANDOM_VARIABLE)

# %%
print(expr4)

# %%
# The following function checks if random variables are defined
# outside an Integrate statement.
wrong_expression = x * x
wrong_expression.check_rv()

# %%
# The same function called from the correct expression returns an
# empty set.
expr4.check_rv()

# %%
# Calculating its value requires the C++ implementation.
expr4.get_value_c(my_data, prepare_ids=True)

# %%
# We illustrate now the Elem function. It takes two arguments: a
# dictionary, and a formula for the key. For each entry in the
# database, the formula is evaluated, and its result identifies which
# formula in the dictionary should be evaluated.  Here is 'Person' is
# 1, the expression is
#
# .. math:: e_1=2 \beta_1 - \frac{\exp(-\beta_2)}{\beta_3 (\beta_2 \geq \beta_1)},
#
# and if 'Person' is 2, the expression is
#
# .. math:: e_2=2 \beta_1 V_1 - \frac{\exp(-\beta_2 V_2) }{ \beta_3 (\beta_2
#     \geq \beta_1)}.
#
# As it is a regular expression, it can be included in any
# formula. Here, we illustrate it by dividing the result by 10.
elemExpr = Elem({1: expr1, 2: expr2}, Person)
expr5 = elemExpr / 10
print(expr5)

# %%
expr5.dict_of_elementary_expression(TypeOfElementaryExpression.VARIABLE)

# %%
# Note that `Variable1` and `Variable2` have previously been involved
# in another formula. Therefore, they have been numbered according to
# this formula, and this numbering is invalid for the new expression
# `expr5`. An error is triggered
try:
    expr5.get_value_c(database=my_data)
except BiogemeError as e:
    print(e)

# %%
expr5.get_value_c(database=my_data, prepare_ids=True)

# %%
testElem = MonteCarlo(Elem({1: my_draws * my_draws}, 1))

# %%
testElem.audit()

# %%
# The next expression is simply the sum of multiple expressions. The
# argument is a list of expressions.
expr6 = bioMultSum([expr1, expr2, expr4])
print(expr6)

# %%
expr6.get_value_c(database=my_data, number_of_draws=NUMBER_OF_DRAWS, prepare_ids=True)

# %%
# We now illustrate how to calculate a logit model, that is
#
# .. math:: \frac{y_1 e^{V_1}}{y_0 e^{V_0}+y_1 e^{V_1}+y_2 e^{V_2}}
#
# where :math:`V_0=-\beta_1`, :math:`V_1=-\beta_2` and
# :math:`V_2=-\beta_1`, and :math:`y_i = 1`, :math:`i=1,2,3`.
V = {0: -beta1, 1: -beta2, 2: -beta1}
av = {0: 1, 1: 1, 2: 1}
expr7 = _bioLogLogit(V, av, 1)

# %%
expr7.get_value()

# %%
# If the alternative is not in the choice set, an exception is raised.
expr7_wrong = LogLogit(V, av, 3)
try:
    expr7_wrong.get_value()
except BiogemeError as e:
    print(f'Exception: {e}')

# %%
# It is actually better to use the C++ implementation, available in
# the module models.

# %%
expr8 = loglogit(V, av, 1)
expr8.get_value_c(database=my_data, prepare_ids=True)

# %%
# As the result is a numpy array, it can be used for any
# calculation. Here, we show how to calculate the logsum.
for v in V.values():
    print(v.get_value_c(database=my_data, prepare_ids=True))

# %%
logsum = np.log(
    np.sum(
        [np.exp(v.get_value_c(database=my_data, prepare_ids=True)) for v in V.values()],
        axis=1,
    )
)
display(logsum)

# %%
# It is possible to calculate the derivative of a formula with respect
# to a literal:
#
# .. math:: e_9=\frac{\partial e_8}{\partial \beta_2}.
expr9 = Derive(expr8, 'beta2')
expr9.get_value_c(database=my_data, prepare_ids=True)

# %%
display(expr9.elementaryName)

# %%
# Biogeme also provides an approximation of the CDF of the normal
# distribution:
#
# .. math:: e_{10}= \frac{1}{{\sigma \sqrt {2\pi } }}\int_{-\infty}^t
#     e^{{{ - \left( {x - \mu } \right)^2 } \mathord{\left/ {\vphantom
#     {{ - \left( {x - \mu } \right)^2 } {2\sigma ^2 }}} \right. }
#     {2\sigma ^2 }}}dx.
expr10 = bioNormalCdf(Variable1 / 10 - 1)
expr10.get_value_c(database=my_data, prepare_ids=True)

# %%
# Min and max operators are also available. To avoid any ambiguity
# with the Python operator, they are called bioMin and bioMax.
expr11 = bioMin(expr5, expr10)
expr11.get_value_c(database=my_data, prepare_ids=True)

# %%
expr12 = bioMax(expr5, expr10)
expr12.get_value_c(database=my_data, prepare_ids=True)

# %%
# For the sake of efficiency, it is possible to specify explicitly a
# linear function, where each term is the product of a parameter and a
# variable.
terms = [
    LinearTermTuple(beta=beta1, x=Variable('Variable1')),
    LinearTermTuple(beta=beta2, x=Variable('Variable2')),
    LinearTermTuple(beta=beta3, x=Variable('newvar_b')),
]
expr13 = bioLinearUtility(terms)
expr13.get_value_c(database=my_data, prepare_ids=True)

# %%
# In terms of specification, it is equivalent to the expression
# below. But the calculation of the derivatives is more efficient, as
# the linear structure of the specification is exploited.
expr13bis = beta1 * Variable1 + beta2 * Variable2 + beta3 * newvar_b
expr13bis.get_value_c(database=my_data, prepare_ids=True)

# %%
# A Pythonic way to write a linear utility function.
variables = ['v1', 'v2', 'v3', 'cost', 'time', 'headway']
coefficients = {f'{v}': Beta(f'beta_{v}', 0, None, None, 0) for v in variables}
terms = [coefficients[v] * Variable(v) for v in variables]
util = sum(terms)
print(util)

# %%
# If the data is organized a panel data, it means that several rows
# correspond to the same individual. The expression
# `PanelLikelihoodTrajectory` calculates the product of the expression
# evaluated for each row. If Monte Carlo integration is involved, the
# same draws are used for each them.

# %%
# Our database contains 5 observations.
my_data.get_sample_size()

# %%
my_data.panel('Person')

# %%
# Once the data has been labeled as "panel", it is considered that
# there are only two series of observations, corresponding to each
# person. Each of these observations is associated with several rows
# of observations.

# %%
my_data.get_sample_size()

# %%
# If we try to evaluate again the integral :math:`\int_0^1 x^2
# dx=\frac{1}{3}`, an exception is raised.
try:
    expr3.get_value_c(database=my_data)
except BiogemeError as e:
    print(f'Exception: {e}')

# %%
# This is detected by the `audit` function, called before the
# expression is evaluated.
expr3.audit(database=my_data)

# %%
# We now evaluate an expression for panel data.
c1 = bioDraws('draws1', 'NORMAL_HALTON2')
c2 = bioDraws('draws2', 'NORMAL_HALTON2')
U1 = Beta('beta1', 0, None, None, 0) * Variable1 + 10 * c1
U2 = Beta('beta2', 0, None, None, 0) * Variable2 + 10 * c2
U3 = 0
U = {1: U1, 2: U2, 3: U3}
av = {1: Av1, 2: Av2, 3: Av3}
expr14 = log(MonteCarlo(PanelLikelihoodTrajectory(logit(U, av, Choice))))

# %%
expr14.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
display(expr14)

# %%
expr14.get_value_c(database=my_data, number_of_draws=NUMBER_OF_DRAWS, prepare_ids=True)

# %%
expr14.get_value_and_derivatives(
    database=my_data,
    number_of_draws=NUMBER_OF_DRAWS,
    gradient=True,
    hessian=True,
    aggregation=False,
)

# %%
expr14.get_value_and_derivatives(
    database=my_data,
    number_of_draws=NUMBER_OF_DRAWS,
    gradient=True,
    hessian=True,
    aggregation=True,
)

# %%
# A Python function can also be obtained for this expression. Note
# that it is available only for the aggregated version, summing over
# the database.

# %%
the_function = expr14.create_function(
    database=my_data, number_of_draws=NUMBER_OF_DRAWS, gradient=True, hessian=True
)

# %%
the_function([0, 0])

# %%
the_function([0.1, 0.1])

# %%
# It is possible to fix the value of some (or all) Beta parameters
print(expr14)

# %%
expr14.fix_betas({'beta2': 0.123})

# %%
print(expr14)

# %%
# The name of the parameter can also be changed while fixing its value.

# %%
expr14.fix_betas({'beta2': 123}, prefix='prefix_', suffix='_suffix')

# %%
print(expr14)

# %%
# It can also be renamed using the following function.

# %%
expr14.rename_elementary(['prefix_beta2_suffix'], prefix='PREFIX_', suffix='_SUFFIX')

# %%
print(expr14)

# %%
# Signatures
# ----------

# %%
# The Python library communicates the expressions to the C++ library
# using a syntax called a "signature". We describe and illustrate now
# the signature for each expression. Each expression is identified by
# an identifier provided by Python using the function 'id'.

# %%
id(expr1)

# %%
# Numerical expression
# ++++++++++++++++++++

# %%
# <Numeric>{identifier},0.0
Numeric(0).get_signature()

# %%
# Beta parameters
# +++++++++++++++

# %%
# <Beta>{identifier}"name"[status],uniqueId,betaId'
# where
#
# - status is 0 for free parameters, and non zero for fixed
#   parameters,
# - uniqueId is a unique index given by Biogeme to all elementary
#   expressions,
# - betaId is a unique index given by Biogeme to all free parameters,
#   and to all fixed parameters.

# %%
# As the signature requires an Id, we need to prepare the expression
# first.

# %%
beta1.prepare(database=my_data, number_of_draws=0)
beta1.get_signature()

# %%
beta3.prepare(database=my_data, number_of_draws=0)
beta3.get_signature()

# %%
# Variables
# +++++++++

# %%
# <Variable>{identifier}"name",uniqueId,variableId
# where
#
# - uniqueId is a unique index given by Biogeme to all elementary
#   expressions,
# - variableId is a unique index given by Biogeme to all variables.

# %%
Variable1.get_signature()

# %%
# Random variables
# ++++++++++++++++

# %%
# <RandomVariable>{identifier}"name",uniqueId,randomVariableId
# where
#
# - uniqueId is a unique index given by Biogeme to all elementary
#   expressions,
# - randomVariableId is a unique index given by Biogeme to all random
#   variables.

# %%
omega.prepare(database=my_data, number_of_draws=0)
omega.get_signature()

# %%
# Draws
# +++++

# %%
# <bioDraws>{identifier}"name",uniqueId,drawId
# where
#
# - uniqueId is a unique index given by Biogeme to all elementary
#   expressions,
# - drawId is a unique index given by Biogeme to all draws.

# %%
my_draws.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
my_draws.get_signature()

# %%
# General expression
# ++++++++++++++++++
# `<operator>{identifier}(numberOfChildren),idFirstChild,idSecondChild,idThirdChild,`
# etc...
#
# where the number of identifiers given after the comma matches the
# reported number of children.
#
# Specific examples are reported below.

# %%
# Binary operator
# ///////////////

# %%
# `<code><operator>{identifier}(2),idFirstChild,idSecondChild </code>`
# where operator is one of:
#
#     - `Plus`
#     - `Minus`
#     - `Times`
#     - `Divide`
#     - `Power`
#     - `bioMin`
#     - `bioMax`
#     - `And`
#     - `Or`
#     - `Equal`
#     - `NotEqual`
#     - `LessOrEqual`
#     - `GreaterOrEqual`
#     - `Less`
#     - `Greater`

# %%
the_sum = beta1 + Variable1

# %%
the_sum.get_signature()

# %%
# Unary operator
# //////////////

# %%
# `<operator>{identifier}(1),idChild,`
# where operator is one of:
#
#     - `UnaryMinus`
#     - `MonteCarlo`
#     - `bioNormalCdf`
#     - `PanelLikelihoodTrajectory`
#     - `exp`
#     - `log`

# %%
m = -beta1

# %%
m.get_signature()

# %%
# LogLogit
# ////////

# %%
# <LogLogit>{identifier}(nbrOfAlternatives),
# chosenAlt,altNumber,utility,availability,altNumber,utility,availability, etc.

# %%
expr7.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
expr7.get_signature()

# %%
# Derive
# //////

# %%
# <Derive>{identifier},id of expression to derive,unique index of elementary expression

# %%
expr9.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
expr9.get_signature()

# %%
# Integrate
# /////////

# %%
# <Integrate>{identifier},id of expression to derive,index of random variable

# %%
expr4.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
expr4.get_signature()

# %%
# Elem
# ////

# %%
# <Elem>{identifier}(number_of_expressions),keyId,value1,expression1,value2,expression2, etc...
#
# where
#
# - keyId is the identifier of the expression calculating the key,
# - the number of pairs valuex,expressionx must correspond to the
#   value of number_of_expressions

# %%
elemExpr.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
elemExpr.get_signature()

# %%
# bioLinearUtility
# ////////////////

# %%
# <bioLinearUtility>{identifier}(numberOfTerms),
# beta1_exprId, beta1_uniqueId, beta1_name, variable1_exprId, variable1_uniqueId, variable1_name, etc...
#
# where 6 entries are provided for each term:
#
#     - beta1_exprId is the expression id of the Beta parameter
#     - beta1_uniqueId is the unique id of the Beta parameter
#     - beta1_name is the name of the parameter
#     - variable1_exprId is the expression id of the variable
#     - variable1_uniqueId is the unique id of the variable
#     - variable1_name is the name of the variable

# %%
expr13.prepare(database=my_data, number_of_draws=NUMBER_OF_DRAWS)
expr13.get_signature()
