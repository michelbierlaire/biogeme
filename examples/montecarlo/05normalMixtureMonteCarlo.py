"""File: 05normalMixtureMonteCarlo.py

 Author: Michel Bierlaire, EPFL
 Date: Wed Dec 11 17:11:45 2019

Calculation of a mixtures of logit models where the integral is
calculated using numerical integration and Monte-Carlo integration
with various types of draws.

"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.draws as draws
import biogeme.distributions as dist
import biogeme.models as models

from biogeme.expressions import Beta, DefineVariable, RandomVariable, Integrate, MonteCarlo, bioDraws 

p = pd.read_csv("swissmetro.dat",sep='\t')
# Use only the first observation
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
B_TIME_RND_normal = B_TIME + B_TIME_S * \
                    bioDraws('B_NORMAL','NORMAL')
B_TIME_RND_anti = B_TIME + B_TIME_S * \
                  bioDraws('B_ANTI','NORMAL_ANTI')
B_TIME_RND_halton = B_TIME + B_TIME_S * \
                    bioDraws('B_HALTON','NORMAL_HALTON2')
B_TIME_RND_mlhs = B_TIME + B_TIME_S * bioDraws('B_MLHS','NORMAL_MLHS')
B_TIME_RND_antimlhs = B_TIME + B_TIME_S * \
                      bioDraws('B_ANTIMLHS','NORMAL_MLHS_ANTI')

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

def logit(THE_B_TIME_RND):
    V1 = ASC_TRAIN + \
         THE_B_TIME_RND * TRAIN_TT_SCALED + \
         B_COST * TRAIN_COST_SCALED
    V2 = ASC_SM + \
         THE_B_TIME_RND * SM_TT_SCALED + \
         B_COST * SM_COST_SCALED
    V3 = ASC_CAR + \
         THE_B_TIME_RND * CAR_TT_SCALED + \
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
    return integrand

numericalI = Integrate(logit(B_TIME_RND)*density,'omega')
normal = MonteCarlo(logit(B_TIME_RND_normal))
anti = MonteCarlo(logit(B_TIME_RND_anti))
halton = MonteCarlo(logit(B_TIME_RND_halton))
mlhs = MonteCarlo(logit(B_TIME_RND_mlhs))
antimlhs = MonteCarlo(logit(B_TIME_RND_antimlhs))

simulate = {'Numerical': numericalI,
            'MonteCarlo': normal,
            'Antithetic': anti,
            'Halton': halton,
            'MLHS':mlhs,
            'Antithetic MLHS': antimlhs}

R = 20000
biogeme = bio.BIOGEME(database,simulate,numberOfDraws=R)
results = biogeme.simulate()
print(f'Number of draws: {10*R}')
for c in results.columns:
    print(f'{c}:\t{results.loc[0,c]}')
