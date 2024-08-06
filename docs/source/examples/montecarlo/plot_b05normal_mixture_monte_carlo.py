"""

Monte-Carlo integration
=======================

Calculation of a mixtures of logit models where the integral is
calculated using numerical integration and Monte-Carlo integration
with various types of draws.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 20:58:50 2023
"""

import biogeme.biogeme as bio
from biogeme import models
import biogeme.distributions as dist
from biogeme.expressions import (
    Expression,
    Integrate,
    RandomVariable,
    MonteCarlo,
    bioDraws,
)

from swissmetro_one import (
    database,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    SM_AV,
    CAR_AV_SP,
    CHOICE,
)

# %%
R = 10000

# %%
# Parameters
ASC_CAR = 0.137
ASC_TRAIN = -0.402
ASC_SM = 0
B_TIME = -2.26
B_TIME_S = 1.66
B_COST = -1.29

# %%
# Generate several versions of the error component.
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
b_time_rnd = B_TIME + B_TIME_S * omega
b_time_rnd_normal = B_TIME + B_TIME_S * bioDraws('B_NORMAL', 'NORMAL')
b_time_rnd_anti = B_TIME + B_TIME_S * bioDraws('B_ANTI', 'NORMAL_ANTI')
b_time_rnd_halton = B_TIME + B_TIME_S * bioDraws('B_HALTON', 'NORMAL_HALTON2')
b_time_rnd_mlhs = B_TIME + B_TIME_S * bioDraws('B_MLHS', 'NORMAL_MLHS')
b_time_rnd_antimlhs = B_TIME + B_TIME_S * bioDraws('B_ANTIMLHS', 'NORMAL_MLHS_ANTI')


# %%
def logit(the_b_time_rnd: Expression) -> Expression:
    """
    Calculate the conditional logit model for a given random parameter.

    :param the_b_time_rnd: expression for the random parameter.

    :return: logit model to be integrated.
    """
    v_1 = ASC_TRAIN + the_b_time_rnd * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
    v_2 = ASC_SM + the_b_time_rnd * SM_TT_SCALED + B_COST * SM_COST_SCALED
    v_3 = ASC_CAR + the_b_time_rnd * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    v = {1: v_1, 2: v_2, 3: v_3}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # The choice model is a logit, with availability conditions
    integrand = models.logit(v, av, CHOICE)
    return integrand


# %%
# Generate each integral.
numerical_integral = Integrate(logit(b_time_rnd) * density, 'omega')
normal = MonteCarlo(logit(b_time_rnd_normal))
anti = MonteCarlo(logit(b_time_rnd_anti))
halton = MonteCarlo(logit(b_time_rnd_halton))
mlhs = MonteCarlo(logit(b_time_rnd_mlhs))
antimlhs = MonteCarlo(logit(b_time_rnd_antimlhs))

# %%
simulate = {
    'Numerical': numerical_integral,
    'MonteCarlo': normal,
    'Antithetic': anti,
    'Halton': halton,
    'MLHS': mlhs,
    'Antithetic MLHS': antimlhs,
}

# %%
biosim = bio.BIOGEME(database, simulate, number_of_draws=R)

# %%
results = biosim.simulate(the_beta_values={})
results

# %%
print(f'Number of draws: {R}')
for c in results.columns:
    print(f'{c}:\t{results.loc[0, c]}')
