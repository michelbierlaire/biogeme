"""File b04latent_choice_seq_mc.py

Choice model with the latent variable.
Mixture of logit, with Monte-Carlo integration
Measurement equation for the indicators.
Sequential estimation.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 18:00:05 2023

"""

import sys
import biogeme.logging as blog
import biogeme.biogeme as bio
import biogeme.exceptions as excep
from biogeme import models
import biogeme.results as res
import biogeme.messaging as msg
from biogeme.expressions import (
    Beta,
    bioDraws,
    MonteCarlo,
    exp,
    log,
)

from optima import (
    database,
    age_65_more,
    formulaIncome,
    moreThanOneCar,
    moreThanOneBike,
    individualHouse,
    male,
    haveChildren,
    haveGA,
    highEducation,
    TimePT,
    MarginalCostPT,
    TimeCar,
    CostCarCHF,
    distance_km,
    TripPurpose,
    WaitingTimePT,
    Choice,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b04latent_choice_seq_mc.py')


# Read the estimates from the structural equation estimation
FILENAME = 'b02one_latent_ordered'
try:
    structResults = res.bioResults(pickleFile=f'{FILENAME}.pickle')
except excep.BiogemeError:
    print(
        f'Run first the script {FILENAME}.py in order to generate the '
        f'file {FILENAME}.pickle.'
    )
    sys.exit()
structBetas = structResults.getBetaValues()


# Coefficients

coef_intercept = structBetas['coef_intercept']
coef_age_65_more = structBetas['coef_age_65_more']
coef_haveGA = structBetas['coef_haveGA']
coef_moreThanOneCar = structBetas['coef_moreThanOneCar']
coef_moreThanOneBike = structBetas['coef_moreThanOneBike']
coef_individualHouse = structBetas['coef_individualHouse']
coef_male = structBetas['coef_male']
coef_haveChildren = structBetas['coef_haveChildren']
coef_highEducation = structBetas['coef_highEducation']

# Latent variable: structural equation

# Define a random parameter, normally distributed, designed to be used
# for numerical integration
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
    + sigma_s * bioDraws('EC', 'NORMAL_MLHS')
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
CostCarCHF_scaled = database.DefineVariable('CostCarCHF_scaled', CostCarCHF / 10)
distance_km_scaled = database.DefineVariable('distance_km_scaled', distance_km / 5)
PurpHWH = database.DefineVariable('PurpHWH', TripPurpose == 1)
PurpOther = database.DefineVariable('PurpOther', TripPurpose != 1)

# Definition of utility functions:

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
loglike = log(MonteCarlo(condprob))

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'b04latent_choice_seq_mc'

# Estimate the parameters
results = the_biogeme.estimate()

print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
results.writeLaTeX()
print(f'LaTeX file: {results.data.latexFileName}')
