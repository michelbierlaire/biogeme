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
import biogeme.version as ver
import biogeme.biogeme_logging as blog
import biogeme.database as db
import biogeme.loglikelihood as ll
import biogeme.expressions as ex
import biogeme.models as md

# %%
# Version of Biogeme.
print(ver.getText())

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# This module provides some basic expressions for the contribution of
# an observation to the (log) likelihood function.

# %%
# Let's consider first a simple choice model.

# %%
V1 = 0
beta = ex.Beta('beta', 0, None, None, 0)
sigma = ex.Beta('sigma', 1, 0, None, 0)
V2 = beta + sigma * ex.bioDraws('V2', 'NORMAL')
V = {1: V1, 2: V2}
condprob = md.logit(V, None, 0)
prob = ex.MonteCarlo(condprob)
print(prob)

# %%
# The first function simply takes the log of the probability for each
# observation.

# %%
loglike = ll.loglikelihood(prob)
print(loglike)

# %%
# The second function also involves the integral using Monte-Carlo
# simulation.

# %%
loglike = ll.mixedloglikelihood(condprob)
print(loglike)

# %%
# Regression models are often used in the context of hybrid choice
# models. Consider the following model.

# %%
x = ex.Variable('x')
y = ex.Variable('y')
beta = ex.Beta('beta', 1, None, None, 0)
sigma = ex.Beta('sigma', 1, None, None, 0)
intercept = ex.Beta('intercept', 0, None, None, 0)
model = intercept + beta * x

# %%
# The following function calculates the contribution to the likelihood. It is
#
#  .. math:: \frac{1}{\sigma} \phi\left( \frac{y-m}{\sigma} \right),
#
#  where :math:`\phi(\cdot)` is the pdf of the normal distribution.
like = ll.likelihoodregression(y, model, sigma)
print(like)

# %%
# The following function calculates the log of the contribution to the
# likelihood. It is
#
# .. math:: -\left( \frac{(y-m)^2}{2\sigma^2} \right) -
#               \log(\sigma) - \frac{1}{2}\log(2\pi).
loglike = ll.loglikelihoodregression(y, model, sigma)
print(loglike)

# %%
# We compare the two on a small database.

# %%
df = pd.DataFrame({'x': [-2, -1, 0, 1, 2], 'y': [1, 1, 1, 1, 1]})
my_data = db.Database('test', df)

# %%
lr = like.getValue_c(my_data, prepareIds=True)
lr

# %%
np.log(lr)

# %%
loglike.getValue_c(my_data, prepareIds=True)
