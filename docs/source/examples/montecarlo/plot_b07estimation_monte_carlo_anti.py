"""

Mixtures of logit with Monte-Carlo 2000 antithetic draws
========================================================

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration with antithetic draws.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 22:43:46 2023
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.expressions import bioDraws
from b07estimation_specification import get_biogeme
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07estimation_monte_carlo_anti.py')

# %%
R = 2000

# %%
the_draws = bioDraws('b_time_rnd', 'NORMAL_ANTI')
the_biogeme = get_biogeme(the_draws=the_draws, number_of_draws=R)
the_biogeme.modelName = 'b07estimation_monte_carlo_anti'

# %%
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
