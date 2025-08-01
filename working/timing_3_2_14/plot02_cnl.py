"""

Timing of a cross-nested logit model
====================================

Michel Bierlaire
Tue Jul 2 14:49:25 2024
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
from biogeme.models import logcnl
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from timing_expression import timing_expression

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %% Nest parameters.
MU_EXISTING = Beta('MU_EXISTING', 1.1, 1, 10, 0)
MU_PUBLIC = Beta('MU_PUBLIC', 1.1, 1, 10, 0)

# %%
# Nest membership parameters.
ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.5, 0, 1, 0)
ALPHA_PUBLIC = 1 - ALPHA_EXISTING

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of nests.

nest_existing = OneNestForCrossNestedLogit(
    nest_param=MU_EXISTING,
    dict_of_alpha={1: ALPHA_EXISTING, 2: 0.0, 3: 1.0},
    name='existing',
)

nest_public = OneNestForCrossNestedLogit(
    nest_param=MU_PUBLIC, dict_of_alpha={1: ALPHA_PUBLIC, 2: 1.0, 3: 0.0}, name='public'
)

nests = NestsForCrossNestedLogit(
    choice_set=[1, 2, 3], tuple_of_nests=(nest_existing, nest_public)
)

# %%
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
log_probability: Expression = logcnl(V, av, nests, CHOICE)

# %%
database = read_data()

# %%
# Timing
timing_results = timing_expression(
    the_expression=log_probability, the_database=database
)
results = [[k, f'{v:.3g}'] for k, v in timing_results.items()]
print(tabulate(results, headers=['', 'Time (in sec.)'], tablefmt='github'))
