"""File 04latentChoiceSeq.py

Choice model with the latent variable.
Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

:author: Michel Bierlaire, EPFL
:date: Tue Sep 10 08:13:18 2019

"""

import sys
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.distributions as dist
import biogeme.results as res
import biogeme.messaging as msg
from biogeme.expressions import (
    Beta,
    RandomVariable,
    exp,
    log,
    Integrate,
)

# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# Read the estimates from the structural equation estimation
try:
    structResults = res.bioResults(pickleFile='02oneLatentOrdered.pickle')
except FileNotFoundError:
    print(
        'Run first the script 02oneLatentOrdered.py in order to generate the '
        'file 02oneLatentOrdered.pickle.'
    )
    sys.exit()
structBetas = structResults.getBetaValues()

### Variables

# Piecewise linear definition of income
ScaledIncome = database.DefineVariable('ScaledIncome', CalculatedIncome / 1000)
thresholds = [None, 4, 6, 8, 10, None]
piecewiseVariables = models.piecewiseVariables(ScaledIncome, thresholds)
formulaIncome = (
    structBetas['beta_ScaledIncome_minus_inf_4'] * piecewiseVariables[0]
    + structBetas['beta_ScaledIncome_4_6'] * piecewiseVariables[1]
    + structBetas['beta_ScaledIncome_6_8'] * piecewiseVariables[2]
    + structBetas['beta_ScaledIncome_8_10'] * piecewiseVariables[3]
    + structBetas['beta_ScaledIncome_10_inf'] * piecewiseVariables[4]
)


# Definition of other variables
age_65_more = database.DefineVariable('age_65_more', age >= 65)
moreThanOneCar = database.DefineVariable('moreThanOneCar', NbCar > 1)
moreThanOneBike = database.DefineVariable('moreThanOneBike', NbBicy > 1)
individualHouse = database.DefineVariable('individualHouse', HouseType == 1)
male = database.DefineVariable('male', Gender == 1)
haveChildren = database.DefineVariable(
    'haveChildren', ((FamilSitu == 3) + (FamilSitu == 4)) > 0
)
haveGA = database.DefineVariable('haveGA', GenAbST == 1)
highEducation = database.DefineVariable('highEducation', Education >= 6)

### Coefficients

coef_intercept = structBetas['coef_intercept']
coef_age_65_more = structBetas['coef_age_65_more']
coef_haveGA = structBetas['coef_haveGA']
coef_moreThanOneCar = structBetas['coef_moreThanOneCar']
coef_moreThanOneBike = structBetas['coef_moreThanOneBike']
coef_individualHouse = structBetas['coef_individualHouse']
coef_male = structBetas['coef_male']
coef_haveChildren = structBetas['coef_haveChildren']
coef_highEducation = structBetas['coef_highEducation']

### Latent variable: structural equation

# Define a random parameter, normally distributed, designed to be used
# for numerical integration
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s = Beta('sigma_s', 1, None, None, 0)

CARLOVERS = (
    coef_intercept
    + coef_age_65_more * age_65_more
    + formulaIncome
    + coef_moreThanOneCar * moreThanOneCar
    + coef_moreThanOneBike * moreThanOneBike
    + coef_individualHouse * individualHouse
    + coef_male * male
    + coef_haveChildren * haveChildren
    + coef_haveGA * haveGA
    + coef_highEducation * highEducation
    + sigma_s * omega
)

# Choice model
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0, None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', 0, None, None, 0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', 0, None, None, 0)
BETA_DIST = Beta('BETA_DIST', 0, None, None, 0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF', 0, None, 0, 0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', 0, None, 0, 0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', 0, None, None, 0)

# The coefficient of the latent variable should be initialized to
# something different from zero. If not, the algorithm may be trapped
# in a local optimum, and never change the value.
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL', -0.01, None, None, 0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL', -0.01, None, None, 0)


TimePT_scaled = database.DefineVariable('TimePT_scaled', TimePT / 200)
TimeCar_scaled = database.DefineVariable('TimeCar_scaled', TimeCar / 200)
MarginalCostPT_scaled = database.DefineVariable(
    'MarginalCostPT_scaled', MarginalCostPT / 10
)
CostCarCHF_scaled = database.DefineVariable(
    'CostCarCHF_scaled', CostCarCHF / 10
)
distance_km_scaled = database.DefineVariable(
    'distance_km_scaled', distance_km / 5
)
PurpHWH = database.DefineVariable('PurpHWH', TripPurpose == 1)
PurpOther = database.DefineVariable('PurpOther', TripPurpose != 1)

### Definition of utility functions:

BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_CL * CARLOVERS)

V0 = (
    ASC_PT
    + BETA_TIME_PT * TimePT_scaled
    + BETA_WAITING_TIME * WaitingTimePT
    + BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH
    + BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
)

BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_CL * CARLOVERS)

V1 = (
    ASC_CAR
    + BETA_TIME_CAR * TimeCar_scaled
    + BETA_COST_HWH * CostCarCHF_scaled * PurpHWH
    + BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
)

V2 = ASC_SM + BETA_DIST * distance_km_scaled

# Associate utility functions with the numbering of alternatives
V = {0: V0, 1: V1, 2: V2}

# Conditional to omega, we have a logit model (called the kernel)
condprob = models.logit(V, None, Choice)
# We integrate over omega using numerical integration
loglike = log(Integrate(condprob * density, 'omega'))

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, loglike)
biogeme.modelName = '04latentChoiceSeq'

# Estimate the parameters
results = biogeme.estimate()

print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
results.writeLaTeX()
print(f'LaTeX file: {results.data.latexFileName}')
