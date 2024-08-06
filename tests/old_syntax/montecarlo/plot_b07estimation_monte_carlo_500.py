"""

Mixtures of logit with Monte-Carlo 500 draws
============================================

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 22:42:06 2023
"""

import biogeme.biogeme_logging as blog
from biogeme.expressions import bioDraws
from b07estimation_specification import get_biogeme

# %%
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07estimation_monte_carlo_500.py')

# %%
R = 500

# %%
the_draws = bioDraws('B_TIME_RND', 'NORMAL')
the_biogeme = get_biogeme(the_draws=the_draws, number_of_draws=R)
the_biogeme.modelName = 'b07estimation_monte_carlo_500'

# %%
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
