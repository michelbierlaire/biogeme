"""File gamma.py

:author: Michel Bierlaire, EPFL
:date: Thu Aug 24 14:18:28 2023

Estimation of a MDCEV model with the "gamma_profile" specification.
"""
import numpy as np
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.mdcev import mdcev_gamma
from biogeme.expressions import Beta, exp
from first_spec import (
    database,
    weight,
    number_chosen,
    consumed_quantities,
    baseline_utilities,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example linear.py')


gamma_shopping = Beta('gamma_shopping', 0.5643, 0.001, None, 0)
gamma_socializing = Beta('gamma_socializing', 1.4978, 0.001, None, 0)
gamma_recreation = Beta('gamma_recreation', 2.7701, 0.001, None, 0)
gamma_personal = Beta('gamma_personal', 0.2093, 0.001, None, 0)


gamma_parameters = {
    1: gamma_shopping,
    2: gamma_socializing,
    3: gamma_recreation,
    4: gamma_personal,
}

logprob = mdcev_gamma(
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
    baseline_utilities=baseline_utilities,
    gamma_parameters=gamma_parameters,
)

# Create the Biogeme object
formulas = {'loglike': logprob, 'weight': weight}
the_biogeme = bio.BIOGEME(database, formulas)
the_biogeme.modelName = 'gamma_profile'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.short_summary())

# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
print(pandas_results)

