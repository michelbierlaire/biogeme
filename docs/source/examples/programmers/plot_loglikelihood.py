"""

biogeme.loglikelihood
=====================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

Michel Bierlaire
Sun Jun 29 2025, 10:59:55
"""

# %%
import numpy as np
import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.database import Database
from biogeme.expressions import Beta, Draws, MonteCarlo, Variable
from biogeme.jax_calculator.simple_formula import (
    evaluate_simple_expression_per_row,
)
from biogeme.loglikelihood import (
    likelihoodregression,
    loglikelihood,
    loglikelihoodregression,
    mixedloglikelihood,
)
from biogeme.models import logit
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.version import get_text

# %%
# Version of Biogeme.
print(get_text())

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# This module provides some basic expressions for the contribution of
# an observation to the (log) likelihood function.

# %%
# Let's consider first a simple choice model.

# %%
v_1 = 0
beta = Beta('beta', 0, None, None, 0)
sigma = Beta('sigma', 1, 0, None, 0)
v_2 = beta + sigma * Draws('v2', 'NORMAL')
v = {1: v_1, 2: v_2}
conditional_probability = logit(v, None, 0)
probability = MonteCarlo(conditional_probability)
display(probability)

# %%
# The first function simply takes the log of the probability for each
# observation.

# %%
loglike = loglikelihood(probability)
display(loglike)

# %%
# The second function also involves the integral using Monte-Carlo
# simulation.

# %%
loglike = mixedloglikelihood(conditional_probability)
print(loglike)

# %%
# Regression models are often used in the context of hybrid choice
# models. Consider the following model.

# %%
x = Variable('x')
y = Variable('y')
beta = Beta('beta', 1, None, None, 0)
sigma = Beta('sigma', 1, None, None, 0)
intercept = Beta('intercept', 0, None, None, 0)
model = intercept + beta * x

# %%
# The following function calculates the contribution to the likelihood. It is
#
#  .. math:: \frac{1}{\sigma} \phi\left( \frac{y-m}{\sigma} \right),
#
#  where :math:`\phi(\cdot)` is the pdf of the normal distribution.
like = likelihoodregression(y, model, sigma)
display(like)

# %%
# The following function calculates the log of the contribution to the
# likelihood. It is
#
# .. math:: -\left( \frac{(y-m)^2}{2\sigma^2} \right) -
#               \log(\sigma) - \frac{1}{2}\log(2\pi).
loglike = loglikelihoodregression(y, model, sigma)
print(loglike)

# %%
# We compare the two on a small database.

# %%
df = pd.DataFrame({'x': [-2, -1, 0, 1, 2], 'y': [1, 1, 1, 1, 1]})
my_data = Database('test', df)

# %%
lr = evaluate_simple_expression_per_row(
    expression=like,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Likelihood evaluated by Biogeme: {lr}')

# %%
display(f'Log of the above likelihood: {np.log(lr)}')

# %%
log_lr = evaluate_simple_expression_per_row(
    expression=loglike,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Log likelihood evaluated by Biogeme: {log_lr}')
