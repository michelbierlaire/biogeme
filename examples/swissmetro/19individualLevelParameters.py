"""File 19individualLevelParameters

:author: Michel Bierlaire, EPFL
:date: Wed Aug 26 14:56:49 2020

Calculation of the individual level parameters for model 05normalMixture
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, DefineVariable, bioDraws, MonteCarlo

# Read the data
df = pd.read_csv('swissmetro.dat', '\t')
database = db.Database('swissmetro', df)

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Here we use the "biogeme" way for backward compatibility
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

# Values of the parameters estimated by the model 05normalMixture
ASC_CAR = Beta('ASC_CAR', 0.137, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', -0.402, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', -1.28, None, None, 0)

# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
B_TIME = Beta('B_TIME', -2.26, None, None, 0)

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1.65, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0),
                           database)
TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0),
                             database)
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0,
                                 database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100,
                                   database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,
                              database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,
                                database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,
                               database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100,
                               database)

# Definition of the utility functions
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

# Conditional to B_TIME_RND, we have a logit model (called the kernel)
prob_train = models.logit(V, av, 1)
prob_SM = models.logit(V, av, 2)
prob_car = models.logit(V, av, 3)

numerator_train = MonteCarlo(B_TIME_RND * prob_train)
numerator_SM = MonteCarlo(B_TIME_RND * prob_SM)
numerator_car = MonteCarlo(B_TIME_RND * prob_car)
denominator_train = MonteCarlo(prob_train)
denominator_SM = MonteCarlo(prob_SM)
denominator_car = MonteCarlo(prob_car)

simulate = {'Numerator train': numerator_train,
            'Numerator SM': numerator_SM,
            'Numerator car': numerator_car,
            'Denominator train': denominator_train,
            'Denominator SM': denominator_SM,
            'Denominator car': denominator_car}

biosim = bio.BIOGEME(database, simulate, numberOfDraws=100000)
sim = biosim.simulate()

beta_hat_train = sim['Numerator train'] / sim['Denominator train']
beta_hat_SM = sim['Numerator SM'] / sim['Denominator SM']
beta_hat_car = sim['Numerator car'] / sim['Denominator car']

print(f'Mean of time parameter for individuals choosing '
      f'train: {beta_hat_train}')
print(f'Mean of time parameter for individuals choosing '
      f'SM: {beta_hat_SM}')
print(f'Mean of time parameter for individuals choosing '
      f'car: {beta_hat_car}')
