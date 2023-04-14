""" File: b07estimation_monte_carlo_halton_500.py

 Author: Michel Bierlaire, EPFL
 Date: 1

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration.

"""

import biogeme.logging as blog
from biogeme.expressions import bioDraws
from b07estimation_specification import get_biogeme

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b07estimation_monte_carlo_halton_500.py')

R = 500
the_draws = bioDraws('B_TIME_RND', 'NORMAL_HALTON2')
the_biogeme = get_biogeme(the_draws)
the_biogeme.number_of_draws = R
the_biogeme.modelName = 'b07estimation_monte_carlo_halton_500'
results = the_biogeme.estimate()

# Get the results in a pandas table
print(results.shortSummary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
