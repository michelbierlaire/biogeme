"""

10. Nested logit model normalized from bottom
=============================================

Bayesian estimation of a nested logit model where the normalization is done at the
 bottom level.

Michel Bierlaire, EPFL
Mon Nov 03 2025, 20:07:02
"""

from pathlib import Path

from IPython.core.display_functions import display

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

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import (
    BayesianResults,
    BayesianResultsSummary,
    get_pandas_estimated_parameters,
)
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import lognested_mev_mu
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b10nested_bottom.py')

# %%
# The scale parameters must stay away from zero. We define a small but positive lower bound
POSITIVE_LOWER_BOUND = 1.0e-5

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, 0, 0)
b_cost = Beta('b_cost', 0, None, 0, 0)

# %%
# This is the scale parameter of the choice model. It is usually
# normalized to one. In this example, we normalize the nest parameter
# instead, and estimate the scale parameter for the model.
scale_parameter = Beta('scale_parameter', 0.5, POSITIVE_LOWER_BOUND, 1.0, 0)

# %%
# Definition of the utility functions
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
# Definition of nests. Only the non trivial nests must be defined. A
# trivial nest is a nest containing exactly one alternative. The nest parameter is normalized to 1.
nest_parameter = 1.0
existing = OneNestForNestedLogit(
    nest_param=nest_parameter, list_of_alternatives=[1, 3], name='existing'
)

nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(existing,))

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
# The choice model is a nested logit, with availability conditions,
# where the scale parameter mu is explicitly involved.
log_probability = lognested_mev_mu(v, av, nests, CHOICE, scale_parameter)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b10_nested_bottom'

# %%
# Estimate the posterior distribution of the parameters, or read the results if
# already available.
yaml_file = Path('saved_results') / f'{the_biogeme.model_name}.yaml'
try:
    summary_results = BayesianResultsSummary.from_yaml_file(filename=yaml_file)
except FileNotFoundError:
    results: BayesianResults = the_biogeme.bayesian_estimation()
    summary_results = results.to_summary()

# %%
print(summary_results.short_summary())

# %%
# Present the parameter estimates in a pandas table.
pandas_results = get_pandas_estimated_parameters(
    estimation_results=summary_results,
)
display(pandas_results)

# %%
# Report the variables stored in the Bayesian estimation results.
display(summary_results.report_stored_variables())
