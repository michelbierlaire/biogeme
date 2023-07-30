""" Estimate the parameters of the logit models with the generated sample

:author: Michel Bierlaire
:date: Sat Apr 15 16:32:36 2023
"""

import biogeme.biogeme as bio
from biogeme import models

from sample import SAMPLE_SIZE as N_ALT
from choice_data import database

from model import V, log_probability

# Logit model, with correction for importance sampling of alternatives
logprob = models.loglogit_sampling(V, None, log_probability, 0)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'logit'

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood({i: 1 for i in range(N_ALT)})

# Estimate the parameters
results = the_biogeme.estimate(recycle=True)
print(results.shortSummary())
