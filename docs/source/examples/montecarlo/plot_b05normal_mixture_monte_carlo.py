"""

Monte-Carlo integration
=======================

Calculation of a mixtures of logit models where the integral is
calculated using numerical integration and Monte-Carlo integration
with various types of draws.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 21:11:02
"""

from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Draws,
    Expression,
    IntegrateNormal,
    MonteCarlo,
    RandomVariable,
)
from biogeme.models import logit

from swissmetro_one import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

# %%
R = 2_000_000

# %%
# Parameters
asc_car = 0.137
asc_train = -0.402
asc_sm = 0
b_time = -2.26
b_time_s = 1.66
b_cost = -1.29

# %%
# Generate several versions of the error component.
omega = RandomVariable('omega')
b_time_rnd = b_time + b_time_s * omega
b_time_rnd_normal = b_time + b_time_s * Draws('b_normal', 'NORMAL')
b_time_rnd_anti = b_time + b_time_s * Draws('b_anti', 'NORMAL_ANTI')
b_time_rnd_halton = b_time + b_time_s * Draws('b_halton', 'NORMAL_HALTON2')
b_time_rnd_mlhs = b_time + b_time_s * Draws('b_mlhs', 'NORMAL_MLHS')
b_time_rnd_antimlhs = b_time + b_time_s * Draws('b_antimlhs', 'NORMAL_MLHS_ANTI')


# %%
def conditional_logit(the_b_time_rnd: Expression) -> Expression:
    """
    Calculate the conditional logit model for a given random parameter.

    :param the_b_time_rnd: expression for the random parameter.

    :return: logit model to be integrated.
    """
    v_train = asc_train + the_b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
    v_swissmetro = asc_sm + the_b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
    v_car = asc_car + the_b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    v = {1: v_train, 2: v_swissmetro, 3: v_car}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # The choice model is a logit, with availability conditions
    integrand = logit(v, av, CHOICE)
    return integrand


# %%
# Generate each integral.
numerical_integral = IntegrateNormal(conditional_logit(b_time_rnd), 'omega')
normal = MonteCarlo(conditional_logit(b_time_rnd_normal))
anti = MonteCarlo(conditional_logit(b_time_rnd_anti))
halton = MonteCarlo(conditional_logit(b_time_rnd_halton))
mlhs = MonteCarlo(conditional_logit(b_time_rnd_mlhs))
antimlhs = MonteCarlo(conditional_logit(b_time_rnd_antimlhs))

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
biosim = BIOGEME(database, simulate, number_of_draws=R)

# %%
results = biosim.simulate(the_beta_values={})
display(results)

# %%
print(f'Number of draws: {R}')
for c in results.columns:
    print(f'{c}:\t{results.loc[0, c]}')
