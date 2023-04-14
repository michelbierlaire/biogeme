"""File m03_simultaneous_estimation.py

Choice model with the latent variable.
Mixture of logit.
Measurement equation for the indicators.
Maximum likelihood (full information) estimation.

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 10:07:43 2023
"""

import sys
import biogeme.logging as blog
import biogeme.biogeme as bio
import biogeme.distributions as dist
import biogeme.results as res
from biogeme import models
from biogeme.expressions import (
    Beta,
    Variable,
    log,
    RandomVariable,
    Integrate,
    Elem,
    bioNormalCdf,
    exp,
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
logger.info('Example m03_simultaneous_estimation.py')


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
struct_betas = structResults.getBetaValues()


# Coefficients

coef_intercept = Beta('coef_intercept', struct_betas['coef_intercept'], None, None, 0)
coef_age_30_less = Beta(
    'coef_age_30_less', struct_betas['coef_age_30_less'], None, None, 0
)
coef_haveChildren = Beta(
    'coef_haveChildren', struct_betas['coef_haveChildren'], None, None, 0
)

coef_highEducation = Beta(
    'coef_highEducation', struct_betas['coef_highEducation'], None, None, 0
)

coef_artisans = Beta('coef_artisans', struct_betas['coef_artisans'], None, None, 0)

coef_employees = Beta('coef_employees', struct_betas['coef_employees'], None, None, 0)

coef_male = Beta('coef_male', struct_betas['coef_male'], None, None, 0)

coef_child_center = Beta(
    'coef_child_center', struct_betas['coef_child_center'], None, None, 0
)
coef_child_suburb = Beta(
    'coef_child_suburb', struct_betas['coef_child_suburb'], None, None, 0
)

# Latent variable: structural equation

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

# Measurement equations

indicators = [
    'ResidCh01',
    'ResidCh04',
    'ResidCh05',
    'ResidCh06',
    'LifSty07',
    'LifSty10',
]

# We define the intercept parameters. The first one is normalized to 0.
INTER = {
    k: Beta(f'INTER_{k}', struct_betas[f'INTER_{k}'], None, None, 0)
    for k in indicators[1:]
}
INTER[indicators[0]] = Beta(f'INTER_{indicators[0]}', 0, None, None, 1)

# We define the coefficients. The first one is normalized to 1.
B = {k: Beta(f'B_{k}', struct_betas[f'B_{k}'], None, None, 0) for k in indicators[1:]}
B[indicators[0]] = Beta(f'B_{indicators[0]}', 1, None, None, 1)

# We define the measurement equations for each indicator
MODEL = {k: INTER[k] + B[k] * ACTIVELIFE for k in indicators}

# We define the scale parameters of the error terms.
SIGMA_STAR = {
    k: Beta(f'SIGMA_STAR_{k}', struct_betas[f'SIGMA_STAR_{k}'], 1.0e-5, None, 0)
    for k in indicators[1:]
}
SIGMA_STAR[indicators[0]] = Beta(f'SIGMA_STAR_{indicators[0]}', 1, None, None, 1)

delta_1 = Beta('delta_1', struct_betas['delta_1'], 1.0e-5, None, 0)
delta_2 = Beta('delta_2', struct_betas['delta_2'], 1.0e-5, None, 0)
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1
tau_3 = delta_1
tau_4 = delta_1 + delta_2

tau_1_residual = {k: (tau_1 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
tau_2_residual = {k: (tau_2 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
tau_3_residual = {k: (tau_3 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
tau_4_residual = {k: (tau_4 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
Ind = {
    k: {
        1: bioNormalCdf(tau_1_residual[k]),
        2: bioNormalCdf(tau_2_residual[k]) - bioNormalCdf(tau_1_residual[k]),
        3: bioNormalCdf(tau_3_residual[k]) - bioNormalCdf(tau_2_residual[k]),
        4: bioNormalCdf(tau_4_residual[k]) - bioNormalCdf(tau_3_residual[k]),
        5: 1 - bioNormalCdf(tau_4_residual[k]),
        6: 1.0,
        -1: 1.0,
        -2: 1.0,
    }
    for k in indicators
}

prob_indicators = Elem(Ind[indicators[0]], Variable(indicators[0]))
for k in indicators[1:]:
    prob_indicators *= Elem(Ind[k], Variable(k))

# Choice model
# Read the estimates from the sequential estimation, and use
# them as starting values

NAME_SEQ = 'm02_sequential_estimation'
try:
    choiceResults = res.bioResults(pickleFile=f'{NAME_SEQ}.pickle')
except FileNotFoundError:
    print(
        f'Run first the script {NAME_SEQ}.py in order to generate the file '
        f'{NAME_SEQ}.pickle.'
    )
    sys.exit()

choiceBetas = choiceResults.getBetaValues()

ASC_CAR = Beta('ASC_CAR', choiceBetas['ASC_CAR'], None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', choiceBetas['ASC_SM'], None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', choiceBetas['BETA_COST_HWH'], None, None, 0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', choiceBetas['BETA_COST_OTHER'], None, None, 0)
BETA_DIST = Beta('BETA_DIST', choiceBetas['BETA_DIST'], None, None, 0)
BETA_TIME_CAR_REF = Beta(
    'BETA_TIME_CAR_REF', choiceBetas['BETA_TIME_CAR_REF'], None, 0, 0
)
BETA_TIME_CAR_AL = Beta(
    'BETA_TIME_CAR_AL', choiceBetas['BETA_TIME_CAR_AL'], None, None, 0
)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', choiceBetas['BETA_TIME_PT_REF'], None, 0, 0)
BETA_TIME_PT_AL = Beta('BETA_TIME_PT_AL', choiceBetas['BETA_TIME_PT_AL'], None, None, 0)
BETA_WAITING_TIME = Beta(
    'BETA_WAITING_TIME', choiceBetas['BETA_WAITING_TIME'], None, None, 0
)

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

# Conditional to omega, we have a logit model (called the kernel) for
# the choice
condprob = models.logit(V, None, Choice)

# Conditional to omega, we have the product of ordered probit for the
# indicators.
condlike = prob_indicators * condprob

# We integrate over omega using numerical integration
loglike = log(Integrate(condlike * density, 'omega'))


# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'm03_simultaneous_estimation'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.shortSummary())
print(results.getEstimatedParameters())

print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
