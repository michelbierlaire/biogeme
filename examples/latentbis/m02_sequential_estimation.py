"""File m02_sequential_estimation.py

Choice model with the latent variable.
Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 09:47:53 2023

"""

import sys
import biogeme.logging as blog
import biogeme.exceptions as excep
import biogeme.biogeme as bio
import biogeme.distributions as dist
import biogeme.results as res
from biogeme import models
from biogeme.expressions import (
    Beta,
    RandomVariable,
    exp,
    log,
    Integrate,
)

from optima import (
    database,
    male,
    age,
    haveChildren,
    highEducation,
    childCenter,
    childSuburb,
    SocioProfCat,
    TimePT,
    TimeCar,
    MarginalCostPT,
    CostCarCHF,
    distance_km,
    TripPurpose,
    WaitingTimePT,
    Choice,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example m02_sequential_estimation.py')

# Read the estimates from the structural equation estimation
NAME = 'm01_latent_variable'
try:
    structResults = res.bioResults(pickleFile=f'{NAME}.pickle')
except excep.BiogemeError:
    print(
        f'Run first the script {NAME}.py in order to generate the file '
        f'{NAME}.pickle.'
    )
    sys.exit()
struct_betas = structResults.getBetaValues()

# Coefficients

coef_intercept = struct_betas['coef_intercept']
coef_age_30_less = struct_betas['coef_age_30_less']
coef_male = struct_betas['coef_male']
coef_haveChildren = struct_betas['coef_haveChildren']
coef_highEducation = struct_betas['coef_highEducation']
coef_artisans = struct_betas['coef_artisans']
coef_employees = struct_betas['coef_employees']
coef_child_center = struct_betas['coef_child_center']
coef_child_suburb = struct_betas['coef_child_suburb']

# Latent variable: structural equation

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
CostCarCHF_scaled = database.DefineVariable('CostCarCHF_scaled', CostCarCHF / 10)
distance_km_scaled = database.DefineVariable('distance_km_scaled', distance_km / 5)
PurpHWH = database.DefineVariable('PurpHWH', TripPurpose == 1)
PurpOther = database.DefineVariable('PurpOther', TripPurpose != 1)

# Definition of utility functions:

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

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'm02_sequential_estimation'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.shortSummary())
print(results.getEstimatedParameters())

print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
