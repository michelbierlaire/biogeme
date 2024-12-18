"""

Simulation of a mixture model
=============================

Simulation of the mixture model, with estimation of the integration error.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:47:42 2023

"""

import sys
import numpy as np

from biogeme.biogeme import BIOGEME
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, bioDraws, MonteCarlo
from biogeme.models import logit
from biogeme.results_processing import EstimationResults

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    database,
    CHOICE,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
)

try:
    import matplotlib.pyplot as plt

    PLOT = True
except ModuleNotFoundError:
    print('Install matplotlib to see the distribution of integration errors.')
    print('pip install matplotlib')
    PLOT = False

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
B_TIME = Beta('B_TIME', 0, None, None, 0)

# %%
# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL')

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# The estimation results are read from the pickle file.
try:
    results = EstimationResults.from_yaml_file(
        filename='saved_results/b05normal_mixture.yaml'
    )
except FileNotFoundError:
    print(
        'Run first the script 05normalMixture.py in order to generate the '
        'file b05normal_mixture.yaml.'
    )
    sys.exit()
# %%
# Conditional to b_time_rnd, we have a logit model (called the kernel)
prob = logit(V, av, CHOICE)

# %%
# We calculate the integration error. Note that this formula assumes
# independent draws, and is not valid for Halton or antithetic draws.
integral = MonteCarlo(prob)
integralSquare = MonteCarlo(prob * prob)
variance = integralSquare - integral * integral
error = (variance / 2.0) ** 0.5

# %%
# And the value of the individual parameters.
numerator = MonteCarlo(B_TIME_RND * prob)
denominator = integral

# %%
simulate = {
    'Numerator': numerator,
    'Denominator': denominator,
    'Integral': integral,
    'Integration error': error,
}

# %%
# Create the Biogeme object.
biosim = BIOGEME(database, simulate, number_or_draws=100)
biosim.modelName = 'b05normal_mixture_simul'

# %%
# NUmber of draws
print(biosim.number_of_draws)

# %%
# Simulate the requested quantities. The output is a Pandas data frame.
simresults = biosim.simulate(results.get_beta_values())

# %%
# 95% confidence interval on the log likelihood.
simresults['left'] = np.log(
    simresults['Integral'] - 1.96 * simresults['Integration error']
)
simresults['right'] = np.log(
    simresults['Integral'] + 1.96 * simresults['Integration error']
)

# %%
print(f'Log likelihood: {np.log(simresults["Integral"]).sum()}')

# %%
print(
    f'Integration error for {biosim.number_of_draws} draws: '
    f'{simresults["Integration error"].sum()}'
)

# %%
print(f'In average {simresults["Integration error"].mean()} per observation.')

# %%
# 95% confidence interval
print(
    f'95% confidence interval: [{simresults["left"].sum()} - '
    f'{simresults["right"].sum()}]'
)

# %%
# Post processing to obtain the individual parameters.
simresults['Beta'] = simresults['Numerator'] / simresults['Denominator']

# %%
# Plot the histogram of individual parameters
if PLOT:
    simresults['Beta'].plot(kind='hist', density=True, bins=20)


# %%
# Plot the general distribution of Beta
def normalpdf(val: float, mu: float = 0.0, std: float = 1.0) -> float:
    """
    Calculate the pdf of the normal distribution, for plotting purposes.

    """
    d = -(val - mu) * (val - mu)
    n = 2.0 * std * std
    a = d / n
    num = np.exp(a)
    den = std * 2.506628275
    p = num / den
    return p


# %%
betas = results.get_beta_values(['B_TIME', 'B_TIME_S'])
x = np.arange(simresults['Beta'].min(), simresults['Beta'].max(), 0.01)

# %%
if PLOT:
    plt.plot(x, normalpdf(x, betas['B_TIME'], betas['B_TIME_S']), '-')
    plt.show()
