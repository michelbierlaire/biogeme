"""

biogeme.cnl
===========

Example of usage of the cnl module.  This is for programmers who need
examples of use of the functions of the class. The examples are
designed to illustrate the syntax.

Michel Bierlaire
Sun Jun 29 2025, 02:25:16

"""

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.cnl import cnl_cdf, cnl_g
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.tools import CheckDerivativesResults, check_derivatives

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Logging on')

# %%
# Definition of the nests.
choice_set = [1, 2, 3, 4]
mu_nest_1 = 1.4
alphas_1 = {1: 1, 2: 0.5, 3: 0.2}
nest_1 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_1, dict_of_alpha=alphas_1, name='Nest 1'
)
mu_nest_2 = 1.2
alphas_2 = {2: 0.5, 3: 0.8, 4: 1}
nest_2 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_2, dict_of_alpha=alphas_2, name='Nest 2'
)
nests = NestsForCrossNestedLogit(choice_set=choice_set, tuple_of_nests=(nest_1, nest_2))

# %%
# We retrieve the G function of the cross-nested logit, and verify
# numerically the implementation of the derivatives.

# %%
G = cnl_g(choice_set, nests)

# %%
# Draw a random point where to evaluate the function.
y = np.random.uniform(low=0.01, high=2, size=4)
display(y)

# %%
check_results: CheckDerivativesResults = check_derivatives(G, y, names=None, logg=True)

# %%
print(f'f = {check_results.function}')
# %%
# We display the differences between the entries of the analytical gradient and the finite differences gradient
display(pd.DataFrame(check_results.errors_gradient))
# %%
# We display the differences between the entries of the analytical hessian and the finite differences hessian
display(pd.DataFrame(check_results.errors_hessian))

# %%
# We do the same for the CDF.

# %%
xi = np.random.uniform(low=-10, high=10, size=4)
display(xi)

# %%
F = cnl_cdf(choice_set, nests)

# %%
check_cdf_results: CheckDerivativesResults = check_derivatives(
    F, y, names=None, logg=True
)
# %%
print(f'f = {check_cdf_results.function}')
# %%
# We display the differences between the entries of the analytical gradient and the finite differences gradient
display(pd.DataFrame(check_cdf_results.errors_gradient))
# %%
# We display the differences between the entries of the analytical hessian and the finite differences hessian
display(pd.DataFrame(check_cdf_results.errors_hessian))
