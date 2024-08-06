"""

Mixtures of logit with Monte-Carlo 2000 Halton draws
====================================================

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration with Halton draws.

:author: Michel Bierlaire, EPFL
:date: Mon Dec 11 08:10:40 2023
"""

import biogeme.biogeme_logging as blog
from biogeme.expressions import bioDraws
from b07estimation_specification import get_biogeme

# %%
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07estimation_monte_carlo_halton.py')

# %%
R = 2000

# %%
the_draws = bioDraws('B_TIME_RND', 'NORMAL_HALTON2')
the_biogeme = get_biogeme(the_draws=the_draws, number_of_draws=R)
the_biogeme.modelName = 'b07estimation_monte_carlo_halton'

# %%
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
