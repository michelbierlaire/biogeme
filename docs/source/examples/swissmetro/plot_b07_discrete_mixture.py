"""

7. Latent class model
=====================

Example of a discrete mixture of logit (or latent class model).

Michel Bierlaire, EPFL
Sat Jun 21 2025, 15:11:24

"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, log
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

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
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Class membership probability.
prob_class1 = Beta('prob_class1', 0.5, 0, 1, 0)
prob_class2 = 1 - prob_class1

# %%
# Definition of the utility functions for latent class 1, where the
# time coefficient is zero.
v_train_class_1 = asc_train + b_cost * TRAIN_COST_SCALED
v_swissmetro_class_1 = asc_sm + b_cost * SM_COST_SCALED
v_car_class_1 = asc_car + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v_class_1 = {1: v_train_class_1, 2: v_swissmetro_class_1, 3: v_car_class_1}

# %%
# Definition of the utility functions for latent class 2, where the
# time coefficient is estimated.
v_train_class_2 = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro_class_2 = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car_class_2 = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v_class_2 = {1: v_train_class_2, 2: v_swissmetro_class_2, 3: v_car_class_2}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# The choice model is a discrete mixture of logit, with availability conditions
choice_probability_class_1 = logit(v_class_1, av, CHOICE)
choice_probability_class_2 = logit(v_class_2, av, CHOICE)
prob = (
    prob_class1 * choice_probability_class_1 + prob_class2 * choice_probability_class_2
)
log_probability = log(prob)

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b07_discrete_mixture'

# %%
# Estimate the parameters.
try:
    results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{the_biogeme.model_name}.yaml'
    )
except FileNotFoundError:
    results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
