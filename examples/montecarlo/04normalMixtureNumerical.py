""" File: 04normalMixtureNumerical.py

 Author: Michel Bierlaire, EPFL
 Date: Wed Dec 11 17:06:52 2019

Calculation of a mixtures of logit models where the integral is
calculated using numerical integration.
"""

# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.distributions as dist
from biogeme import models

from biogeme.expressions import RandomVariable, Integrate

p = pd.read_csv('swissmetro.dat', sep='\t')
# Use only the first observation (index 0)
p = p.drop(p[p.index != 0].index)
database = db.Database('swissmetro', p)

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Parameters
ASC_CAR = 0.137
ASC_TRAIN = -0.402
ASC_SM = 0
B_TIME = -2.26
B_TIME_S = 1.66
B_COST = -1.29

# Define a random parameter, normally distributed,
# designed to be used for integration
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
B_TIME_RND = B_TIME + B_TIME_S * omega

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)
CAR_AV_SP = CAR_AV * (SP != 0)
TRAIN_AV_SP = TRAIN_AV * (SP != 0)
TRAIN_TT_SCALED = TRAIN_TT / 100.0
TRAIN_COST_SCALED = TRAIN_COST / 100
SM_TT_SCALED = SM_TT / 100.0
SM_COST_SCALED = SM_COST / 100
CAR_TT_SCALED = CAR_TT / 100
CAR_CO_SCALED = CAR_CO / 100

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# The choice model is a logit, with availability conditions
integrand = models.logit(V, av, CHOICE)
numericalI = Integrate(integrand * density, 'omega')

simulate = {'Numerical': numericalI}

biogeme = bio.BIOGEME(database, simulate)
results = biogeme.simulate()
print(
    'Mixture of logit - numerical integration: ', results.iloc[0]['Numerical']
)
