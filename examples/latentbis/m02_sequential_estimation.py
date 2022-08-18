"""File m02_sequential_estimation.py

Choice model with the latent variable.
Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

:author: Michel Bierlaire, EPFL
:date: Tue Jul  6 18:04:50 2021

"""

import sys
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.distributions as dist
import biogeme.results as res
import biogeme.messaging as msg
import biogeme.optimization as opt
from biogeme import models
from biogeme.expressions import (
    Beta,
    Variable,
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
Choice = Variable('Choice')
Gender = Variable('Gender')
FamilSitu = Variable('FamilSitu')
Education = Variable('Education')
ResidChild = Variable('ResidChild')
SocioProfCat = Variable('SocioProfCat')
age = Variable('age')
TimePT = Variable('TimePT')
TimeCar = Variable('TimeCar')
MarginalCostPT = Variable('MarginalCostPT')
CostCarCHF = Variable('CostCarCHF')
distance_km = Variable('distance_km')
TripPurpose = Variable('TripPurpose')
WaitingTimePT = Variable('WaitingTimePT')
Choice = Variable('Choice')

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# Read the estimates from the structural equation estimation
NAME = 'm01_latent_variable'
try:
    structResults = res.bioResults(pickleFile=f'{NAME}.pickle')
except FileNotFoundError:
    print(
        f'Run first the script {NAME}.py in order to generate the file '
        f'{NAME}.pickle.'
    )
    sys.exit()
structBetas = structResults.getBetaValues()

### Variables

# Definition of other variables
male = database.DefineVariable('male', Gender == 1)

haveChildren = database.DefineVariable(
    'haveChildren', ((FamilSitu == 3) + (FamilSitu == 4)) > 0
)

highEducation = database.DefineVariable('highEducation', Education >= 6)

childCenter = database.DefineVariable(
    'childCenter', ((ResidChild == 1) + (ResidChild == 2)) > 0
)

childSuburb = database.DefineVariable(
    'childSuburb', ((ResidChild == 3) + (ResidChild == 4)) > 0
)


### Coefficients

coef_intercept = structBetas['coef_intercept']
coef_age_30_less = structBetas['coef_age_30_less']
coef_male = structBetas['coef_male']
coef_haveChildren = structBetas['coef_haveChildren']
coef_highEducation = structBetas['coef_highEducation']
coef_artisans = structBetas['coef_artisans']
coef_employees = structBetas['coef_employees']
coef_child_center = structBetas['coef_child_center']
coef_child_suburb = structBetas['coef_child_suburb']

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to
# write it across several lines, each line must terminate with
# the \ symbol

# Define a random parameter, normally distributed, designed to be used
# for numerical integration
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s = Beta('sigma_s', 1, None, None, 0)

ACTIVELIFE = (
    coef_intercept
    + coef_child_center * childCenter
    + coef_child_suburb * childSuburb
    + coef_highEducation * highEducation
    + coef_artisans * (SocioProfCat == 5)
    + coef_employees * (SocioProfCat == 6)
    + coef_age_30_less * (age <= 30)
    + coef_male * male
    + coef_haveChildren * haveChildren
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
BETA_TIME_PT_AL = Beta('BETA_TIME_PT_AL', 1, None, None, 0)
BETA_TIME_CAR_AL = Beta('BETA_TIME_CAR_AL', -1, None, None, 0)


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

BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_AL * ACTIVELIFE)

V0 = (
    ASC_PT
    + BETA_TIME_PT * TimePT_scaled
    + BETA_WAITING_TIME * WaitingTimePT
    + BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH
    + BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
)

BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_AL * ACTIVELIFE)

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
biogeme.modelName = 'm02_sequential_estimation'

# Estimate the parameters
results = biogeme.estimate(algorithm=opt.bioBfgs)
print(results.getEstimatedParameters())

print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
