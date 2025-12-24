"""

2. Estimation with weights: WESML
=================================

 Example of a logit model with Weighted Exogenous Sample Maximum
 Likelihood (WESML).

Michel Bierlaire, EPFL
Wed Jun 18 2025, 11:20:51
"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit
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
    GROUP,
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
#
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Definition of the utility functions.
#
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
#
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
#
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
#
#
# This is the contribution of each observation to the log likelihood function.
logprob = loglogit(v, av, CHOICE)

# %%
# Definition of the weight.

WEIGHT_GROUP_2 = 8.890991e-01
WEIGHT_GROUP_3 = 1.2
weight = WEIGHT_GROUP_2 * (GROUP == 2) + WEIGHT_GROUP_3 * (GROUP == 3)

# %%
# These notes will be included as such in the report file.
USER_NOTES = (
    'Example of a logit model with three alternatives: '
    'Train, Car and Swissmetro.'
    ' Weighted Exogenous Sample Maximum Likelihood estimator (WESML)'
)


# %%
#  Create the Biogeme object. Here, we need to provide both the formula for the log likelihood function, and
# the formula for the weights. This is done via a dict with keys `log_like` and `weight`.
# It is possible to control the generation of the HTML and the yaml
# files. Note that these parameters can also be modified in the .TOML
# configuration file.
formulas = {'log_like': logprob, 'weight': weight}
the_biogeme = BIOGEME(
    database, formulas, user_notes=USER_NOTES, generate_html=True, generate_yaml=False
)
the_biogeme.model_name = 'b02_weight'

# %%
# Estimate the parameters.
try:
    results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{the_biogeme.model_name}.yaml'
    )
except FileNotFoundError:
    results = the_biogeme.estimate()

# %%
#
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
