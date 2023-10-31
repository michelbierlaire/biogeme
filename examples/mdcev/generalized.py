"""File linear.py

:author: Michel Bierlaire, EPFL
:date: Thu Aug 24 14:18:28 2023

Estimation of a MDCEV model with the "linear expenditure system" specification.
"""
import numpy as np
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
import numpy as np
from biogeme.mdcev import mdcev_generalized
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


def zero_one_transform(alpha):
    return 1.0 / (1 + exp(alpha))


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

psi_shopping = Beta('psi_shopping', -10, None, None, 0)
psi_socializing = Beta('psi_socializing', -10, None, None, 0)
psi_recreation = Beta('psi_recreation', -10, None, None, 0)
psi_personal = Beta('psi_personal', -10, None, None, 1)

psi_parameters = {
    1: exp(psi_shopping),
    2: exp(psi_socializing),
    3: exp(psi_recreation),
    4: exp(psi_personal),
}


logprob = mdcev_generalized(
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
    baseline_utilities=baseline_utilities,
    alpha_parameters=alpha_parameters,
    gamma_parameters=gamma_parameters,
    psi_parameters=psi_parameters,
)

# Create the Biogeme object
formulas = {'loglike': logprob, 'weight': weight}
the_biogeme = bio.BIOGEME(database, formulas)
the_biogeme.modelName = 'generalized'

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

def numpy_zero_one_transform(alpha):
    return 1.0 / (1 + np.exp(alpha))

for alpha in alphas:
    print(f'{alpha} = {numpy_zero_one_transform(pandas_results.loc[alpha, "Value"]):.3g}')

psi_names = [
    'psi_recreation',
    'psi_shopping',
    'psi_socializing',
]

for psi in psi_names:
    print(f'{psi} = {np.exp(pandas_results.loc[psi, "Value"])}')
    
