"""

18b. Ordinal probit model
=========================

Example of an ordinal probit model.  This is just to illustrate the
syntax, as the data are not ordered.  But the example assume, for the
sake of it, that the alternatives are ordered as 1->2->3

Michel Bierlaire, EPFL
Thu Jun 26 2025, 15:54:37
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, OrderedLogProbit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import CHOICE, TRAIN_COST_SCALED, TRAIN_TT_SCALED, database

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b18b_ordinal_probit.py')

# %%
# Parameters to be estimated
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Threshold parameters for the ordered probit.

# %%
# :math:`\tau_1 \leq 0`.
tau1 = Beta('tau1', -1, None, 0, 0)

# %%
# :math:`\delta_2 \geq 0`.
delta2 = Beta('delta2', 2, 0, None, 0)

# %%
# :math:`\tau_2 = \tau_1 + \delta_2`
tau2 = tau1 + delta2

# %%
# Utility
utility = b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED

# %%
# Associate each discrete indicator with an interval.
log_probability = OrderedLogProbit(
    eta=utility,
    cutpoints=[tau1, tau2],
    y=CHOICE,
    categories=[1, 2, 3],
    neutral_labels=[],
)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b18b_ordinal_probit'

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
