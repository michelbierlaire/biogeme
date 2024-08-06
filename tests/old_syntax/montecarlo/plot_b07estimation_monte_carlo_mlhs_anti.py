"""

Mixtures of logit with Monte-Carlo 2000 antithetic MLHS draws
=============================================================

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration with antithetic MLHS draws.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 23:38:58 2023
"""

import biogeme.biogeme_logging as blog
from biogeme.expressions import bioDraws
from b07estimation_specification import get_biogeme

# %%
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07estimation_specification_mlhs_anti.py')

# %%
R = 2000

# %%
the_draws = bioDraws('B_TIME_RND', 'NORMAL_MLHS_ANTI')
the_biogeme = get_biogeme(the_draws=the_draws, number_of_draws=R)
the_biogeme.modelName = 'b07estimation_monte_carlo_mlhs_anti'

# %%
results = the_biogeme.estimate()

# %%
print(results.shortSummary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
