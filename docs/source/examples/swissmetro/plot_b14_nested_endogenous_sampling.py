"""

14. Nested logit with corrections for endogeneous sampling
==========================================================

The sample is said to be endogenous if the probability for an
individual to be in the sample depends on the choice that has been
made. In that case, the ESML estimator is not appropriate anymore, and
corrections need to be made. See `Bierlaire, Bolduc, McFadden (2008)
<https://dx.doi.org/10.1016/j.trb.2007.09.003>`_.

This is illustrated in this example.

Michel Bierlaire, EPFL
Sat Jun 21 2025, 17:13:33
"""

import numpy as np
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import get_mev_for_nested, logmev_endogenous_sampling
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b14_nested_endogenous_sampling.py')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)
nest_parameter = Beta('nest_parameter', 1, 1, 10, 0)

# %%
# In this example, we assume that the three modes exist, and that the
# sampling protocol is choice-based. The probability that a respondent
# belongs to the sample is R_i.
R_TRAIN = 4.42e-2
R_SM = 3.36e-3
R_CAR = 7.5e-3

# %%
# The correction terms are the log of these quantities
correction = {1: np.log(R_TRAIN), 2: np.log(R_SM), 3: np.log(R_CAR)}

# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of nests. Only the non-trivial nests must be defined. A
# trivial nest is a nest containing exactly one alternative.  In this
# example, we create a nest for the existing modes, that is train (1)
# and car (3).

existing = OneNestForNestedLogit(
    nest_param=nest_parameter, list_of_alternatives=[1, 3], name='existing'
)

nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(existing,))

# %%
# The choice model is a nested logit, with corrections for endogenous sampling
# We first obtain the expression of the Gi function for nested logit.
probability_generating_function = get_mev_for_nested(v, av, nests)

# %%
# Then we calculate the MEV log probability, accounting for the correction.
log_probability = logmev_endogenous_sampling(
    v, probability_generating_function, av, correction, CHOICE
)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b14_nested_endogenous_sampling'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
