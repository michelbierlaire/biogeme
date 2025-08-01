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
from biogeme.expressions import Beta, Expression
from biogeme.models import loglogit
from timing_expression import timing_expression

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)


# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
log_probability: Expression = loglogit(v, av, CHOICE)

#
# %%
database = read_data()

# %%
# Timing
timing_results = timing_expression(
    the_expression=log_probability, the_database=database
)
results = [[k, f'{v:.3g}'] for k, v in timing_results.items()]
print(tabulate(results, headers=['', 'Time (in sec.)'], tablefmt='github'))
