"""

Timing of a logit model
=======================

Michel Bierlaire
Tue Jul 2 14:48:52 2024
"""

from tabulate import tabulate

# %%
# See the data processing script: :ref:`swissmetro_data`.
from biogeme.data.swissmetro import (
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
    read_data,
)
from biogeme.expressions import Beta, MonteCarlo, bioDraws, log
from biogeme.models import logit
from timing_expression import timing_expression

# %%
# Parameters to be estimated
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
b_time = Beta('b_time', 0, None, None, 0)

# %%
# It is advised *not* to use 0 as starting value for the following parameter.
b_time_s = Beta('b_time_s', 1, None, None, 0)
b_time_rnd = b_time + b_time_s * bioDraws('b_time_rnd', 'NORMAL')

# %%
# Definition of the utility functions.
v_train = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional to b_time_rnd, we have a logit model (called the kernel).
prob = logit(V, av, CHOICE)

# %%
# We integrate over b_time_rnd using Monte-Carlo.
log_probability = log(MonteCarlo(prob))

# %%
database = read_data()

# %%
# Number of draws
number_of_draws = 100


# %%
# Timing
timing_results = timing_expression(
    the_expression=log_probability,
    the_database=database,
    number_of_draws=number_of_draws,
)
results = [[k, f'{v:.3g}'] for k, v in timing_results.items()]
print(f'With {number_of_draws} draws...')
print(tabulate(results, headers=['', 'Time (in sec.)'], tablefmt='github'))
