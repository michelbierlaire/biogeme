"""

biogeme.loglikelihood
=====================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Wed Nov 22 15:16:56 2023
"""

# %%
import numpy as np
import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.database import Database
from biogeme.expressions import Beta, bioDraws, MonteCarlo, Variable
from biogeme.loglikelihood import (
    loglikelihood,
    mixedloglikelihood,
    likelihoodregression,
    loglikelihoodregression,
)
from biogeme.models import logit
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
V1 = 0
beta = Beta('Beta', 0, None, None, 0)
sigma = Beta('sigma', 1, 0, None, 0)
V2 = beta + sigma * bioDraws('v2', 'NORMAL')
V = {1: V1, 2: V2}
condprob = logit(V, None, 0)
prob = MonteCarlo(condprob)
print(prob)

# %%
# The first function simply takes the log of the probability for each
# observation.

# %%
loglike = loglikelihood(prob)
print(loglike)

# %%
# The second function also involves the integral using Monte-Carlo
# simulation.

# %%
loglike = mixedloglikelihood(condprob)
print(loglike)

# %%
# Regression models are often used in the context of hybrid choice
# models. Consider the following model.

# %%
x = Variable('x')
y = Variable('y')
beta = Beta('Beta', 1, None, None, 0)
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
print(like)

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
lr = like.get_value_c(my_data, prepare_ids=True)
display(lr)

# %%
np.log(lr)

# %%
loglike.get_value_c(my_data, prepare_ids=True)
