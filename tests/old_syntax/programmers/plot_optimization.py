"""

biogeme.optimization
====================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Thu Nov 23 22:25:13 2023
"""


import pandas as pd
from biogeme.version import getText
import biogeme.biogeme as bio
import biogeme.database as db
from biogeme import models
from biogeme.expressions import Beta, Variable
import biogeme.biogeme_logging as blog


# %%
# Version of Biogeme.
print(getText())

# %%
logger = blog.get_screen_logger(blog.INFO)

# %%
# Data,
df = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [1, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)
df

# %%
my_data = db.Database('test', df)

# %%
# Variables.
Choice = Variable('Choice')
Variable1 = Variable('Variable1')
Variable2 = Variable('Variable2')
beta1 = Beta('beta1', 0, None, None, 0)
beta2 = Beta('beta2', 0, None, None, 0)
V1 = beta1 * Variable1
V2 = beta2 * Variable2
V3 = 0
V = {1: V1, 2: V2, 3: V3}

# %%
likelihood = models.loglogit(V, av=None, i=Choice)
my_biogeme = bio.BIOGEME(my_data, likelihood)
my_biogeme.modelName = 'simpleExample'
my_biogeme.save_iterations = False
my_biogeme.generate_html = False
my_biogeme.generate_pickle = False
print(my_biogeme)

# %%
f, g, h, gdiff, hdiff = my_biogeme.checkDerivatives(
    beta=my_biogeme.beta_values_dict_to_list(), verbose=True
)

# %%
pd.DataFrame(gdiff)

# %%
pd.DataFrame(hdiff)

# %%
# **scipy**: this is the optimization algorithm from scipy.
my_biogeme.algorithm_name = 'scipy'
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# **Newton with linesearch**
my_biogeme.algorithm_name = 'LS-newton'
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# Changing the requested precision
my_biogeme.tolerance = 0.1
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# **Newton with trust region**
my_biogeme.algorithm_name = 'TR-newton'
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# We illustrate the parameters. We use the truncated conjugate
# gradient instead of dogleg for the trust region subproblem, starting
# with a small trust region of radius 0.001, and a maximum of 3
# iterations.

# %%
my_biogeme.dogleg = False
my_biogeme.initial_radius = 0.001
my_biogeme.maxiter = 3
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# Changing the requested precision
my_biogeme.tolerance = 0.1
my_biogeme.maxiter = 1000
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# **BFGS with line search**
my_biogeme.algorithm_name = 'LS-BFGS'
my_biogeme.tolerance = 1.0e-6
my_biogeme.maxiter = 1000
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# **BFGS with trust region**
my_biogeme.algorithm_name = 'TR-BFGS'
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# **Newton/BFGS with trust region for simple bounds**

# %%
# This is the default algorithm used by Biogeme. It is the
# implementation of the algorithm proposed by `Conn et al. (1988)
# <https://www.ams.org/journals/mcom/1988-50-182/S0025-5718-1988-0929544-3/S0025-5718-1988-0929544-3.pdf>`_.
my_biogeme.algorithm_name = 'simple_bounds'
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# When the second derivatives are too computationally expensive to
# calculate, it is possible to avoid calculating them at each
# successful iteration. The parameter `second_derivatives` allows to
# control that.

# %%
my_biogeme.second_derivatives = 0.5
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# If the parameter is set to zero, the second derivatives are not used
# at all, and the algorithm relies only on the BFGS update.

# %%
my_biogeme.second_derivatives = 0.0
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
# There are shortcuts to call the BFGS and the Newton versions
my_biogeme.algorithm_name = 'simple_bounds_newton'
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')

# %%
my_biogeme.algorithm_name = 'simple_bounds_BFGS'
results = my_biogeme.estimate()
results.getEstimatedParameters()

# %%
for k, v in results.data.optimizationMessages.items():
    print(f'{k}:\t{v}')
