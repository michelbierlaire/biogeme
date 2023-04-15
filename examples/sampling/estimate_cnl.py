""" Estimate the parameters of the cross-nested logit models with the generated samples

:author: Michel Bierlaire
:date: Sat Apr 15 16:59:43 2023
"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from biogeme.sampling import mev_cnl_sampling

from sample import SAMPLE_SIZE as N_ALT
from choice_data import database

from model import V, log_probability

mu_asian = Beta('mu_asian', 1, 1, None, 0)
mu_downtown = Beta('mu_downtown', 1, 1, None, 0)

# The nests definition is done through the variables in the file. For
# each nest m and each alternative i, there must be a variable m_i
# that is the level of membership of alternative i to nest m (usually
# called the "alpha" membership parameters).

# In this example, we define two nests, one for Asian restaurants, and
# one for restaurants in Downtown. Note that their overlap, so that we
# indeed have a cross-nested logit model.
nests = {'Asian': mu_asian, 'downtown': mu_downtown}

log_gi = mev_cnl_sampling(V, None, log_probability, nests)
logprob = models.logmev(V, log_gi, None, 0)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'cnl'

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(N_ALT)})

# Estimate the parameters
results = the_biogeme.estimate(recycle=True)
print(results.shortSummary())
