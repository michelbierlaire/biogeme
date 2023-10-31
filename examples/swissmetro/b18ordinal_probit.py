"""File b18ordinal_probit.py

:author: Michel Bierlaire, EPFL
:date: Mon Apr 10 12:15:28 2023

 Example of an ordinal probit model.
 This is just to illustrate the syntax, as the data are not ordered.
 But the example assume, for the sake of it, that they are 1->2->3
"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
import biogeme.distributions as dist
from biogeme.models import ordered_probit
from biogeme.expressions import Beta, log, Elem
from swissmetro_data import (
    database,
    CHOICE,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b18ordinal_probit.py')

# Parameters to be estimated
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Parameters for the ordered probit.
# tau1 <= 0
tau1 = Beta('tau1', -1, None, 0, 0)
# delta2 >= 0
delta2 = Beta('delta2', 2, 0, None, 0)
tau2 = tau1 + delta2

#  Utility
U = B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED

# Associate each discrete indicator with an interval.
#   1: -infinity -> tau1
#   2: tau1 -> tau2
#   3: tau2 -> +infinity

ChoiceProba = {
    1: 1 - dist.logisticcdf(U - tau1),
    2: dist.logisticcdf(U - tau1) - dist.logisticcdf(U - tau2),
    3: dist.logisticcdf(U - tau2),
}

the_proba = ordered_probit(
    continuous_value=U,
    list_of_discrete_values=[1, 2, 3],
    tau_parameter=tau1,
)

the_chosen_proba = Elem(the_proba, CHOICE)

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = log(the_chosen_proba)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b18ordinal_probit'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.short_summary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
