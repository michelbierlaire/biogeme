"""

11b. Simulation of a cross-nested logit model
=============================================

Illustration of the application of an estimated CNL model.

Michel Bierlaire, EPFL
Sat Jun 21 2025, 16:53:57
"""

import sys

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Derive
from biogeme.models import cnl
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT,
    GA,
    SM_AV,
    SM_COST_SCALED,
    SM_HE,
    SM_TT,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_HE,
    TRAIN_TT,
    database,
)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time_swissmetro = Beta('b_time_swissmetro', 0, None, None, 0)
b_time_train = Beta('b_time_train', 0, None, None, 0)
b_time_car = Beta('b_time_car', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)
b_headway_swissmetro = Beta('b_headway_swissmetro', 0, None, None, 0)
b_headway_train = Beta('b_headway_train', 0, None, None, 0)
ga_train = Beta('ga_train', 0, None, None, 0)
ga_swissmetro = Beta('ga_swissmetro', 0, None, None, 0)

# %% Nest parameters.
existing_nest_parameter = Beta('existing_nest_parameter', 1, 1, 5, 0)
public_nest_parameter = Beta('public_nest_parameter', 1, 1, 5, 0)

# %%
# Nest membership parameters.
alpha_existing = Beta('alpha_existing', 0.5, 0, 1, 0)
alpha_public = 1 - alpha_existing

# %%
# Definition of the utility functions. Note that in order to calculate the derivative with respect to the travel
# time variables, they need to explicitly appear in the specification. Therefore, we have replaced the scaled
# versions of the variables by their original definition.
v_train = (
    asc_train
    + b_time_train * TRAIN_TT / 100
    + b_cost * TRAIN_COST_SCALED
    + b_headway_train * TRAIN_HE
    + ga_train * GA
)
v_swissmetro = (
    asc_sm
    + b_time_swissmetro * SM_TT / 100
    + b_cost * SM_COST_SCALED
    + b_headway_swissmetro * SM_HE
    + ga_swissmetro * GA
)
v_car = asc_car + b_time_car * CAR_TT / 100 + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of nests.

nest_existing = OneNestForCrossNestedLogit(
    nest_param=existing_nest_parameter,
    dict_of_alpha={1: alpha_existing, 2: 0.0, 3: 1.0},
    name='existing',
)

nest_public = OneNestForCrossNestedLogit(
    nest_param=public_nest_parameter,
    dict_of_alpha={1: alpha_public, 2: 1.0, 3: 0.0},
    name='public',
)

nests = NestsForCrossNestedLogit(
    choice_set=[1, 2, 3], tuple_of_nests=(nest_existing, nest_public)
)

# %%
# Read the estimation results from the pickle file.
try:
    results = EstimationResults.from_yaml_file(filename='saved_results/b11cnl.yaml')
except FileNotFoundError:
    print(
        'Run first the script b11cnl.py in order to generate the file b11cnl.yaml, and move it to the directory '
        'saved_results.'
    )
    sys.exit()


# %%
print(
    'Estimation results: ', get_pandas_estimated_parameters(estimation_results=results)
)

# %%
print('Calculating correlation matrix. It may generate numerical warnings from scipy.')
corr = nests.correlation(
    parameters=results.get_beta_values(),
    alternatives_names={1: 'Train', 2: 'Swissmetro', 3: 'Car'},
)
display(corr)

# %%
# The choice model is a cross-nested logit, with availability conditions.
probability_train = cnl(v, av, nests, 1)
probability_swissmetro = cnl(v, av, nests, 2)
probability_car = cnl(v, av, nests, 3)

# %%
# We calculate elasticities. It is important that the variables
# explicitly appear as such in the model. If not, the derivative will
# be zero, as well as the elasticities.
general_time_eslaticity_train = (
    Derive(probability_train, 'TRAIN_TT') * TRAIN_TT / probability_train
)
general_time_elasticity_swissmetro = (
    Derive(probability_swissmetro, 'SM_TT') * SM_TT / probability_swissmetro
)
general_time_elasticity_car = (
    Derive(probability_car, 'CAR_TT') * CAR_TT / probability_car
)

# %%
# We report the probability of each alternative and the elasticities.
simulate = {
    'Prob. train': probability_train,
    'Prob. Swissmetro': probability_swissmetro,
    'Prob. car': probability_car,
    'Elas. 1': general_time_eslaticity_train,
    'Elas. 2': general_time_elasticity_swissmetro,
    'Elas. 3': general_time_elasticity_car,
}

# %%
# Create the Biogeme object.
biosim = BIOGEME(database, simulate)
biosim.model_name = 'b11b_cnl_simul'

# %%
# Perform the simulation.
simulation_results = biosim.simulate(results.get_beta_values())

# %%
print('Simulation results')
display(simulation_results)

# %%
print(
    f'Aggregate share of train: {100 * simulation_results["Prob. train"].mean():.1f}%'
)
