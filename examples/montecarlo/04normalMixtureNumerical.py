""" File: 04normalMixtureNumerical.py

 Author: Michel Bierlaire, EPFL
 Date: Wed Dec 11 17:06:52 2019

Calculation of a mixtures of logit models where the integral is
calculated using numerical integration.

"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.draws as draws
import biogeme.distributions as dist
import biogeme.models as models

from biogeme.expressions import Beta, DefineVariable, RandomVariable, Integrate

p = pd.read_csv("swissmetro.dat",sep='\t')
# Use only the first observation (index 0)
p = p.drop(p[p.index != 0].index)
database = db.Database("swissmetro",p)

globals().update(database.variables)

#Parameters 
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

# Utility functions

#If the person has a GA (season ticket) her 
#incremental cost is actually 0 
#rather than the cost value gathered from the
# network data. 
SM_COST =  SM_CO   * (  GA   ==  0  ) 
TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0. 
# A previous estimation with the unscaled data has generated
# parameters around -0.01 for both cost and time. 
# Therefore, time and cost are multipled my 0.01.

TRAIN_TT_SCALED = \
  DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0,database)
TRAIN_COST_SCALED = \
  DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100,database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100,database)
CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),database)
TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),database)

V1 = ASC_TRAIN + \
     B_TIME_RND * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + \
     B_TIME_RND * SM_TT_SCALED + \
     B_COST * SM_COST_SCALED
V3 = ASC_CAR + \
     B_TIME_RND * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3}

# Associate the availability conditions with the alternatives

av = {1: TRAIN_AV_SP,
      2: SM_AV,
      3: CAR_AV_SP}

# The choice model is a logit, with availability conditions
integrand = models.logit(V,av,CHOICE)
numericalI = Integrate(integrand*density,'omega')

simulate = {'Numerical': numericalI}

biogeme = bio.BIOGEME(database,simulate)
results = biogeme.simulate()
print('Mixture of logit - numerical integration: ',results.iloc[0]['Numerical'])
