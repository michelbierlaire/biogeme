"""File 03choiceOnly.py

Choice model with the latent variable.
Mixture of logit.
No measurement equation for the indicators.

:author: Michel Bierlaire, EPFL
:date: Thu Sep  6 15:14:39 2018

"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.distributions as dist
import biogeme.optimization as opt
import biogeme.messaging as msg
from biogeme.expressions import Beta, DefineVariable, \
    RandomVariable, Integrate, exp, log

# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)


# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

### Variables

# Piecewise linear definition of income
ScaledIncome = DefineVariable('ScaledIncome', CalculatedIncome / 1000, database)

thresholds = [None, 4, 6, 8, 10, None]
formulaIncome = models.piecewiseFormula(ScaledIncome,
                                        thresholds,
                                        [0.0, 0.0, 0.0, 0.0, 0.0])

# Definition of other variables
age_65_more = DefineVariable('age_65_more', age >= 65, database)
moreThanOneCar = DefineVariable('moreThanOneCar', NbCar > 1, database)
moreThanOneBike = DefineVariable('moreThanOneBike', NbBicy > 1, database)
individualHouse = DefineVariable('individualHouse', HouseType == 1, database)
male = DefineVariable('male', Gender == 1, database)
haveChildren = DefineVariable('haveChildren', \
                              ((FamilSitu == 3) + (FamilSitu == 4)) > 0, database)
haveGA = DefineVariable('haveGA', GenAbST == 1, database)
highEducation = DefineVariable('highEducation', Education >= 6, database)

# Parameters to be estimated
coef_intercept = Beta('coef_intercept', 0.0, None, None, 1)
coef_age_65_more = Beta('coef_age_65_more', 0.0, None, None, 0)
coef_age_unknown = Beta('coef_age_unknown', 0.0, None, None, 0)
coef_haveGA = Beta('coef_haveGA', 0.0, None, None, 0)
coef_moreThanOneCar = Beta('coef_moreThanOneCar', 0.0, None, None, 0)
coef_moreThanOneBike = Beta('coef_moreThanOneBike', 0.0, None, None, 0)
coef_individualHouse = Beta('coef_individualHouse', 0.0, None, None, 0)
coef_male = Beta('coef_male', 0.0, None, None, 0)
coef_haveChildren = Beta('coef_haveChildren', 0.0, None, None, 0)
coef_highEducation = Beta('coef_highEducation', 0.0, None, None, 0)

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to
# write it across several lines, each line must terminate with
# the \ symbol

# Define a random parameter, normally distributed, designed to be used
# for numerical integration
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s = Beta('sigma_s', 1, None, None, 0)

CARLOVERS = coef_intercept + \
            coef_age_65_more * age_65_more + \
            formulaIncome + \
            coef_moreThanOneCar * moreThanOneCar + \
            coef_moreThanOneBike * moreThanOneBike + \
            coef_individualHouse * individualHouse + \
            coef_male * male + \
            coef_haveChildren * haveChildren + \
            coef_haveGA * haveGA + \
            coef_highEducation * highEducation + \
            sigma_s * omega

# Choice model

ASC_CAR = Beta('ASC_CAR', 0.0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0.0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0.0, None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', 0.0, None, None, 0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', 0.0, None, None, 0)
BETA_DIST = Beta('BETA_DIST', 0.0, None, None, 0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF', 0.0, None, 0, 0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL', 0.0, None, None, 0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', 0.0, None, 0, 0)
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL', -1.0, None, None, 0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', 0.0, None, None, 0)

TimePT_scaled = DefineVariable('TimePT_scaled', TimePT / 200, database)
TimeCar_scaled = DefineVariable('TimeCar_scaled', TimeCar / 200, database)
MarginalCostPT_scaled = DefineVariable('MarginalCostPT_scaled',
                                       MarginalCostPT / 10, database)
CostCarCHF_scaled = DefineVariable('CostCarCHF_scaled',
                                   CostCarCHF / 10, database)
distance_km_scaled = DefineVariable('distance_km_scaled',
                                    distance_km / 5, database)
PurpHWH = DefineVariable('PurpHWH', TripPurpose == 1, database)
PurpOther = DefineVariable('PurpOther', TripPurpose != 1, database)

### Definition of utility functions:

BETA_TIME_PT = BETA_TIME_PT_REF * \
               exp(BETA_TIME_PT_CL * CARLOVERS)

V0 = ASC_PT + \
     BETA_TIME_PT * TimePT_scaled + \
     BETA_WAITING_TIME * WaitingTimePT + \
     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH  +\
     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

BETA_TIME_CAR = BETA_TIME_CAR_REF * \
                exp(BETA_TIME_CAR_CL * CARLOVERS)

V1 = ASC_CAR + \
      BETA_TIME_CAR * TimeCar_scaled + \
      BETA_COST_HWH * CostCarCHF_scaled * PurpHWH  + \
      BETA_COST_OTHER * CostCarCHF_scaled * PurpOther

V2 = ASC_SM + BETA_DIST * distance_km_scaled

# Associate utility functions with the numbering of alternatives
V = {0: V0,
     1: V1,
     2: V2}

# Conditional to omega, we have a logit model (called the kernel)
condprob = models.logit(V, None, Choice)
# We integrate over omega using numerical integration
loglike = log(Integrate(condprob * density, 'omega'))

# Define level of verbosity
logger = msg.bioMessage()
#logger.setSilent()
#logger.setWarning()
logger.setGeneral()
#logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, loglike)
biogeme.modelName = '03choiceOnly'

# Estimate the parameters
results = biogeme.estimate(algorithm=opt.bioNewton)

print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
results.writeLaTeX()
print(f'LaTeX file: {results.data.latexFileName}')
