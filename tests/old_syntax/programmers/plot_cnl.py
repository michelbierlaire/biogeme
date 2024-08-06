"""

biogeme.cnl
===========

Example of usage of the cnl module.  This is for programmers who need
examples of use of the functions of the class. The examples are
designed to illustrate the syntax.

:author: Michel Bierlaire
:date: Fri Nov 17 08:27:24 2023

"""

import numpy as np
import pandas as pd
import biogeme.cnl as cnl
import biogeme.tools as tools
import biogeme.biogeme_logging as blog
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Logging on')

# %%
# Definition of the nests.
choice_set = ['i', 'j', 'k', 'ell']
mu_nest_1 = 1.4
alphas_1 = {'i': 1, 'j': 0.5, 'k': 0.2}
nest_1 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_1, dict_of_alpha=alphas_1, name='Nest 1'
)
mu_nest_2 = 1.2
alphas_2 = {'j': 0.5, 'k': 0.8, 'ell': 1}
nest_2 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_2, dict_of_alpha=alphas_2, name='Nest 2'
)
nests = NestsForCrossNestedLogit(choice_set=choice_set, tuple_of_nests=(nest_1, nest_2))

# %%
# We retrieve the G function of the cross-nested logit, and verify
# numerically the implementation of the derivatives.

# %%
G = cnl.cnl_G(choice_set, nests)

# %%
# Draw a random point where to evaluate the function.
y = np.random.uniform(low=0.01, high=2, size=4)
y

# %%
f, g, h, gdiff, hdiff = tools.checkDerivatives(G, y, names=None, logg=True)
f

# %%
pd.DataFrame(g)

# %%
pd.DataFrame(h)

# %%
pd.DataFrame(gdiff)

# %%
pd.DataFrame(hdiff)


# %%
# We do the same for the CDF.

# %%
xi = np.random.uniform(low=-10, high=10, size=4)
xi

# %%
F = cnl.cnl_CDF(choice_set, nests)

# %%
f, g, h, gdiff, hdiff = tools.checkDerivatives(F, y, names=None, logg=True)
f

# %%
pd.DataFrame(g)

# %%
pd.DataFrame(h)

# %%
pd.DataFrame(gdiff)

# %%
pd.DataFrame(hdiff)
