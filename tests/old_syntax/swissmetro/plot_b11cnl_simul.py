"""

Simulation of a cross-nested logit model
========================================

Illustration of the application of an estimated model.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 18:08:29 2023
"""

import sys
import biogeme.biogeme as bio
from biogeme import models
import biogeme.results as res
from biogeme.expressions import Beta, Derive
import biogeme.exceptions as excep
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    database,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_TT,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_TT,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_TT,
    CAR_CO_SCALED,
)

# %%
# Simulation must be done with estimated value of the
# parameters. We set them to zero here , and read the values from the
# file created after estimation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Nest parameters.
MU_EXISTING = Beta('MU_EXISTING', 1, 1, None, 0)
MU_PUBLIC = Beta('MU_PUBLIC', 1, 1, None, 0)

# %%
# Nest membership parameters.
ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.5, 0, 1, 0)
ALPHA_PUBLIC = 1 - ALPHA_EXISTING

# %%
# Definition of the utility functions. Note that we need to have
# explicitly the unscaled variables if we want to calculate the
# elasticities.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT / 100 + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT / 100 + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT / 100 + B_COST * CAR_CO_SCALED

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
# Read the estimation results from the pickle file.
try:
    results = res.bioResults(pickleFile='saved_results/b11cnl.pickle')
except excep.BiogemeError:
    print('Run first the script b11cnl.py in order to generate the file 11cnl.pickle.')
    sys.exit()

# %%
print('Estimation results: ', results.getEstimatedParameters())

# %%
print('Calculating correlation matrix. It may generate numerical warnings from scipy.')
corr = nests.correlation(
    parameters=results.getBetaValues(),
    alternatives_names={1: 'Train', 2: 'Swissmetro', 3: 'Car'},
)
corr

# %%
# The choice model is a cross-nested logit, with availability conditions.
prob1 = models.cnl_avail(V, av, nests, 1)
prob2 = models.cnl_avail(V, av, nests, 2)
prob3 = models.cnl_avail(V, av, nests, 3)

# %%
# We calculate elasticities. It is important that the variables
# explicitly appear as such in the model. If not, the derivative will
# be zero, as well as the elasticities.
genelas1 = Derive(prob1, 'TRAIN_TT') * TRAIN_TT / prob1
genelas2 = Derive(prob2, 'SM_TT') * SM_TT / prob2
genelas3 = Derive(prob3, 'CAR_TT') * CAR_TT / prob3

# %%
# We report the probability of each alternative and the elasticities.
simulate = {
    'Prob. train': prob1,
    'Prob. Swissmetro': prob2,
    'Prob. car': prob3,
    'Elas. 1': genelas1,
    'Elas. 2': genelas2,
    'Elas. 3': genelas3,
}

# %%
# Create the Biogeme object.
biosim = bio.BIOGEME(database, simulate)
biosim.modelName = 'b11cnl_simul'

# %%
# Perform the simulation.
simresults = biosim.simulate(results.getBetaValues())

# %%
print('Simulation results')
simresults

# %%
print(f'Aggregate share of train: {100*simresults["Prob. train"].mean():.1f}%')
