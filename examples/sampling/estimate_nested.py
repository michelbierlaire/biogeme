""" Estimate the parameters of the nested logit models with the generated samples

:author: Michel Bierlaire
:date: Sat Apr 15 16:45:26 2023
"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from biogeme.sampling import mev_cnl_sampling

from sample import SAMPLE_SIZE as N_ALT
from choice_data import database

from model import V, log_probability

# Nests are defined with indicator variables. In this example, the nest
# "Asian" contains all the alternative i such as "Asian_i" is equal to 1.
# All other alternatives are alone in their own nest.
mu_nested = Beta('mu_nested', 1, 1, None, 0)
nests = {'Asian': mu_nested}

# In this context, the model is actually a cross-nested logit model,
# as the numbering is not consistent across observations. Indeed, if
# restaurant 2 is Asian for one observation, it may not be for the
# next.
log_gi = mev_cnl_sampling(V, None, log_probability, nests)
logprob = models.logmev(V, log_gi, None, 0)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'nested'

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(N_ALT)})

# Estimate the parameters
results = the_biogeme.estimate(recycle=True)
print(results.shortSummary())
