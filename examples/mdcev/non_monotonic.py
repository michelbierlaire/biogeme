"""File linear.py

:author: Michel Bierlaire, EPFL
:date: Thu Aug 24 14:18:28 2023

Estimation of a MDCEV model with the "linear expenditure system" specification.
"""
import numpy as np
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.mdcev import mdcev_non_monotonic
from biogeme.expressions import Beta, exp
from first_spec import (
    database,
    weight,
    number_chosen,
    consumed_quantities,
    baseline_utilities,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example generalized.py')


def zero_one_transform(the_alpha):
    return 1.0 / (1 + exp(the_alpha))


alpha_shopping = Beta('alpha_shopping', 0, 0, None, 0)
alpha_socializing = Beta('alpha_socializing', 0, 0, None, 0)
alpha_recreation = Beta('alpha_recreation', 0, 0, None, 0)
alpha_personal = Beta('alpha_personal', 0, 0, None, 0)

alpha_parameters = {
    1: zero_one_transform(alpha_shopping),
    2: zero_one_transform(alpha_socializing),
    3: zero_one_transform(alpha_recreation),
    4: zero_one_transform(alpha_personal),
}

gamma_shopping = Beta('gamma_shopping', 1, 0.001, None, 0)
gamma_socializing = Beta('gamma_socializing', 1, 0.001, None, 0)
gamma_recreation = Beta('gamma_recreation', 1, 0.001, None, 0)
gamma_personal = Beta('gamma_personal', 1, 0.001, None, 0)


gamma_parameters = {
    1: gamma_shopping,
    2: gamma_socializing,
    3: gamma_recreation,
    4: gamma_personal,
}

logprob = mdcev_non_monotonic(
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
    baseline_utilities=baseline_utilities,
    mu_utilities=baseline_utilities,
    alpha_parameters=alpha_parameters,
    gamma_parameters=gamma_parameters,
)

# Create the Biogeme object
formulas = {'loglike': logprob, 'weight': weight}
the_biogeme = bio.BIOGEME(database, formulas)
the_biogeme.modelName = 'non_monotonic'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.short_summary())

# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
print(pandas_results)

alphas = [
    'alpha_personal',
    'alpha_recreation',
    'alpha_shopping',
    'alpha_socializing',
]

def numpy_zero_one_transform(the_alpha):
    return 1.0 / (1 + np.exp(the_alpha))

for alpha in alphas:
    print(f'{alpha} = {numpy_zero_one_transform(pandas_results.loc[alpha, "Value"]):.3g}')

    
