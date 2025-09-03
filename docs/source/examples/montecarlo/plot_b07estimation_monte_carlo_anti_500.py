"""

Mixtures of logit with Monte-Carlo 500 antithetic draws
=======================================================

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration with antithetic draws.

Michel Bierlaire, EPFL
Sun Jun 29 2025, 06:47:25
"""

from b07estimation_specification import get_biogeme
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.expressions import Draws
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

# %%
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07estimation_monte_carlo_anti_500.py')

# %%
R = 500

# %%
the_draws = Draws('b_time_rnd', 'NORMAL_ANTI')
the_biogeme = get_biogeme(the_draws=the_draws, number_of_draws=R)
the_biogeme.model_name = 'b07estimation_monte_carlo_anti_500'
results_file = f'saved_results/{the_biogeme.model_name}.yaml'


# %%
try:
    results = EstimationResults.from_yaml_file(filename=results_file)
except FileNotFoundError:
    results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
