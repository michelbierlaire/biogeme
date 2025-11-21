"""

19. Calculation of individual level parameters
==============================================

Calculation of the individual level parameters for the model defined
in :ref:`plot_b05normal_mixture`.

Michel Bierlaire, EPFL
Thu Jun 26 2025, 15:55:41

"""

from IPython.core.display_functions import display
from pandas.core.interchange.dataframe_protocol import DataFrame

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Draws, MonteCarlo
from biogeme.models import logit
from biogeme.results_processing import EstimationResults

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

# %%
# Parameters. The initial value is irrelevant.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
b_time = Beta('b_time', 0, None, None, 0)
b_time_s = Beta('b_time_s', 1, None, None, 0)
b_time_rnd = b_time + b_time_s * Draws('b_time_rnd', 'NORMAL')

# %%
# Retrieve estimation results
result_file_name = 'saved_results/b05a_normal_mixture.yaml'
the_estimation_results = EstimationResults.from_yaml_file(filename=result_file_name)

# %%
# Definition of the utility functions.
v_train = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional on b_time_rnd, we have a logit model (called the kernel).
prob_chosen = logit(v, av, CHOICE)

# %%
# Numerator and denominator of the formula for individual parameters.
numerator = MonteCarlo(b_time_rnd * prob_chosen)
denominator = MonteCarlo(prob_chosen)

# %%
simulate = {
    'Numerator': numerator,
    'Denominator': denominator,
    'Choice': CHOICE,
}

# %%
biosim = BIOGEME(database, simulate, number_of_draws=10_000)
sim: DataFrame = biosim.simulate(the_estimation_results.get_beta_values())
sim['Individual-level parameters'] = sim['Numerator'] / sim['Denominator']

display(sim)
