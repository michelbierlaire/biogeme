"""

Mixtures of logit with Monte-Carlo 10_000 Halton draws
======================================================

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration with Halton draws.

Michel Bierlaire, EPFL
Tue Apr 29 2025, 12:15:53
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from b07estimation_specification import get_biogeme
from biogeme.expressions import Draws
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07estimation_monte_carlo_halton.py')

# %%
R = 10_000

# %%
the_draws = Draws('b_time_rnd', 'NORMAL_HALTON2')
the_biogeme = get_biogeme(the_draws=the_draws, number_of_draws=R)
the_biogeme.model_name = 'b07estimation_monte_carlo_halton'

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
