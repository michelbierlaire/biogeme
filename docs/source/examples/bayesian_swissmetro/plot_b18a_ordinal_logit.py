"""

18a. Ordinal logit model
========================

Bayesian estimation of an ordinal logit model.  This is just to illustrate the
syntax, as the data are not ordered.  But the example assume, for the
sake of it, that the alternatives are ordered as 1->2->3

Michel Bierlaire, EPFL
Mon Nov 17 2025, 16:38:41

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import BayesianResults, get_pandas_estimated_parameters
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, OrderedLogLogit
# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import CHOICE, TRAIN_COST_SCALED, TRAIN_TT_SCALED, database

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b18a_ordinal_logit.py')

# %%
# We define a small but positive lower bound
POSITIVE_LOWER_BOUND = 1.0e-5

# %%
# Parameters to be estimated
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Threshold parameters for the ordered logit.

# %%
# :math:`\tau_1 \leq 0`.
tau1 = Beta('tau1', -1, None, 0, 0)

# %%
# :math:`\delta_2 \geq 0`.
delta2 = Beta('delta2', 2, POSITIVE_LOWER_BOUND, None, 0)

# %%
# :math:`\tau_2 = \tau_1 + \delta_2`
tau2 = tau1 + delta2

# %%
#  Utility.
utility = b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED

# %%
# Associate each discrete indicator with an interval.
#
#   1. :math:`-\infty \to \tau_1`,
#   2. :math:`\tau_1 \to \tau_2`,
#   3. :math:`\tau_2 \to +\infty`.

log_probability = OrderedLogLogit(
    eta=utility,
    cutpoints=[tau1, tau2],
    y=CHOICE,
    categories=[1, 2, 3],
    neutral_labels=[],
)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b18a_ordinal_logit'

# %%
# Estimate the parameters.
try:
    results = BayesianResults.from_netcdf(
        filename=f'saved_results/{the_biogeme.model_name}.nc'
    )
except FileNotFoundError:
    results = the_biogeme.bayesian_estimation()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
