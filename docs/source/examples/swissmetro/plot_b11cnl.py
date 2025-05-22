"""

Cross-nested logit
==================

 Example of a cross-nested logit model with two nests:

 - one with existing alternatives (car and train),
 - one with public transportation alternatives (train and Swissmetro)

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 18:06:44 2023

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import logcnl
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    SM_AV,
    SM_COST_SCALED,
    SM_HE,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_HE,
    TRAIN_TT_SCALED,
    database,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b11cnl.py')

0         ASC_TRAIN -0.308325         0.200044       -1.541283    1.232478e-01
1      B_TIME_TRAIN -1.073824         0.141684       -7.579018    3.486100e-14
2            B_COST -0.973715         0.066189      -14.711051    0.000000e+00
3   B_HEADWAY_TRAIN -0.004366         0.000972       -4.491706    7.065500e-06
4          GA_TRAIN  1.142925         0.231598        4.934960    8.016704e-07
5    ALPHA_EXISTING  0.644550         0.172079        3.745658    1.799215e-04
6       MU_EXISTING  1.771265         0.230121        7.697103    1.398881e-14
7         B_TIME_SM -0.991484         0.177872       -5.574145    2.487487e-08
8      B_HEADWAY_SM -0.007724         0.002969       -2.601454    9.282947e-03
9             GA_SM -0.138687         0.161170       -0.860503    3.895120e-01
10          ASC_CAR -0.606300         0.124075       -4.886564    1.026108e-06
11       B_TIME_CAR -0.856971         0.126767       -6.760227    1.377765e-11
12        MU_PUBLIC  1.840582         0.465324        3.955483    7.638015e-05

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME_SM = Beta('B_TIME_SM', 0, None, None, 0)
B_TIME_TRAIN = Beta('B_TIME_TRAIN', 0, None, None, 0)
B_TIME_CAR = Beta('B_TIME_CAR', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
B_HEADWAY_SM = Beta('B_HEADWAY_SM', 0, None, None, 0)
B_HEADWAY_TRAIN = Beta('B_HEADWAY_TRAIN', 0, None, None, 0)
GA_TRAIN = Beta('GA_TRAIN', 0, None, None, 0)
GA_SM = Beta('GA_SM', 0, None, None, 0)
# %% Nest parameters.
MU_EXISTING = Beta('MU_EXISTING', 1, 1, 5, 0)
MU_PUBLIC = Beta('MU_PUBLIC', 1, 1, 5, 0)

# %%
# Nest membership parameters.
ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.5, 0, 1, 0)
ALPHA_PUBLIC = 1 - ALPHA_EXISTING

# %%
# Definition of the utility functions
V1 = (
    ASC_TRAIN
    + B_TIME_TRAIN * TRAIN_TT_SCALED
    + B_COST * TRAIN_COST_SCALED
    + B_HEADWAY_TRAIN * TRAIN_HE
    + GA_TRAIN * GA
)
V2 = (
    ASC_SM
    + B_TIME_SM * SM_TT_SCALED
    + B_COST * SM_COST_SCALED
    + B_HEADWAY_SM * SM_HE
    + GA_SM * GA
)
V3 = ASC_CAR + B_TIME_CAR * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives
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
# The choice model is a cross-nested logit, with availability conditions.
logprob = logcnl(V, av, nests, CHOICE)

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, logprob)
the_biogeme.model_name = 'b11cnl'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
