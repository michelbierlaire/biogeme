"""

Estimation of several models
============================

Example of the estimation of several specifications of the model.

:author: Michel Bierlaire, EPFL
:date: Mon Apr 10 12:19:46 2023

"""

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, log
from biogeme.catalog import Catalog, segmentation_catalogs
from biogeme.models import loglogit
from biogeme.results_processing import compile_estimation_results, pareto_optimal

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
    MALE,
)

# %%
# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_CAR_MALE = Beta('ASC_CAR_MALE', 0, None, None, 0)
ASC_CAR_FEMALE = Beta('ASC_CAR_FEMALE', 0, None, None, 0)
ASC_TRAIN_MALE = Beta('ASC_TRAIN_MALE', 0, None, None, 0)
ASC_TRAIN_FEMALE = Beta('ASC_TRAIN_FEMALE', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
segmentation_gender = database.generate_segmentation(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

# %%
# We define catalogs with two different specifications for the
# ASC_CAR: non segmented, and segmented.
ASC_TRAIN_catalog = ASC_TRAIN
ASC_CAR_catalog = ASC_CAR

# %%
# We now define a catalog  with the log travel time as well as the travel time.

# %%
# First for train
train_tt_catalog = log(TRAIN_TT_SCALED)

# %%
# Then for SM. But we require that the specification is the same as
# train by defining the same controller.
sm_tt_catalog = log(SM_TT_SCALED)
# %%
# Definition of the utility functions with linear cost.
V1 = ASC_TRAIN_catalog + B_TIME * train_tt_catalog + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * sm_tt_catalog + B_COST * SM_COST_SCALED
V3 = ASC_CAR_catalog + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = loglogit(V, av, CHOICE)

# %%
the_biogeme: BIOGEME = BIOGEME(database=database, formulas=logprob)
the_biogeme.modelName = 'm02'

# %%
results = the_biogeme.estimate()
