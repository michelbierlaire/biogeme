"""

Ordinal probit model
====================

Example of an ordinal probit model.  This is just to illustrate the
syntax, as the data are not ordered.  But the example assume, for the
sake of it, that the alternatives are ordered as 1->2->3

:author: Michel Bierlaire, EPFL
:date: Mon Apr 10 12:15:28 2023

"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.models import ordered_probit
from biogeme.expressions import Beta, log, Elem

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    database,
    CHOICE,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b18ordinal_probit.py')

# %%
# Parameters to be estimated
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

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
U = B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED

# %%
# Associate each discrete indicator with an interval.
#
#   1. :math:`-\infty \to \tau_1`,
#   2. :math:`\tau_1 \to \tau_2`,
#   3. :math:`\tau_2 \to +\infty`.
the_proba = ordered_probit(
    continuous_value=U,
    list_of_discrete_values=[1, 2, 3],
    tau_parameter=tau1,
)

# %%
# Extract from the dict the formula associated with the observed choice.
the_chosen_proba = Elem(the_proba, CHOICE)

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = log(the_chosen_proba)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b18ordinal_probit'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.get_estimated_parameters()
pandas_results
