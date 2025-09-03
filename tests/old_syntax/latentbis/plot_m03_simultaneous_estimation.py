"""

Choice model with the latent_old variable: maximum likelihood estimation
====================================================================

Mixture of logit.
Measurement equation for the indicators.
Maximum likelihood (full information) estimation.

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 10:07:43 2023
"""

import sys
from functools import reduce

from optima import (
    Choice,
    CostCarCHF_scaled,
    MarginalCostPT_scaled,
    PurpHWH,
    PurpOther,
    SocioProfCat,
    TimeCar_scaled,
    TimePT_scaled,
    WaitingTimePT,
    age,
    childCenter,
    childSuburb,
    database,
    distance_km_scaled,
    haveChildren,
    highEducation,
    male,
)
from read_or_estimate import read_or_estimate

import biogeme.biogeme as bio
import biogeme.biogeme_logging as blog
import biogeme.distributions as dist
import biogeme.exceptions as excep
import biogeme.results as res
from biogeme import models
from biogeme.expressions import (
    Beta,
    Elem,
    Integrate,
    RandomVariable,
    Variable,
    bioNormalCdf,
    exp,
    log,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example m03_simultaneous_estimation.py')

# %%
# Read the estimates from the structural equation estimation.
MODELNAME = 'm01_latent_variable'
try:
    struct_results = res.bioResults(pickleFile=f'saved_results/{MODELNAME}.pickle')
except excep.BiogemeError:
    print(
        f'Run first the script {MODELNAME}.py in order to generate the '
        f'file {MODELNAME}.pickle, and move it to the directory saved_results'
    )
    sys.exit()
struct_betas = struct_results.getBetaValues()


# %%
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

# %%
# Latent variable: structural equation

# %%
# Define a random parameter, normally distributed, designed to be used
# for numerical integration
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s = Beta('sigma_s', 1, None, None, 0)

# %%
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

# %%
# Measurement equations
indicators = [
    'ResidCh01',
    'ResidCh04',
    'ResidCh05',
    'ResidCh06',
    'LifSty07',
    'LifSty10',
]

# %%
# We define the intercept parameters. The first one is normalized to 0.
inter = {k: Beta(f'inter_{k}', 0, None, None, 0) for k in indicators[1:]}
inter[indicators[0]] = Beta(f'INTER_{indicators[0]}', 0, None, None, 1)

# %%
# We define the coefficients. The first one is normalized to 1.
coefficients = {k: Beta(f'coeff_{k}', 0, None, None, 0) for k in indicators[1:]}
coefficients[indicators[0]] = Beta(f'B_{indicators[0]}', 1, None, None, 1)

# %%
# We define the measurement equations for each indicator
linear_models = {k: inter[k] + coefficients[k] * ACTIVELIFE for k in indicators}

# %%
# We define the scale parameters of the error terms.
sigma_star = {k: Beta(f'sigma_star_{k}', 1, 1.0e-5, None, 0) for k in indicators[1:]}
sigma_star[indicators[0]] = Beta(f'sigma_star_{indicators[0]}', 1, None, None, 1)

# %%
# Symmetric threshold.
delta_1 = Beta('delta_1', 0.1, 1.0e-5, None, 0)
delta_2 = Beta('delta_2', 0.2, 1.0e-5, None, 0)
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1
tau_3 = delta_1
tau_4 = delta_1 + delta_2

# %%
# Ordered probit models.
tau_1_residual = {k: (tau_1 - linear_models[k]) / sigma_star[k] for k in indicators}
tau_2_residual = {k: (tau_2 - linear_models[k]) / sigma_star[k] for k in indicators}
tau_3_residual = {k: (tau_3 - linear_models[k]) / sigma_star[k] for k in indicators}
tau_4_residual = {k: (tau_4 - linear_models[k]) / sigma_star[k] for k in indicators}
dict_prob_indicators = {
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

# %%
# Product of the likelihood of the indicators.
prob_indicators = reduce(
    lambda x, y: x * Elem(dict_prob_indicators[y], Variable(y)),
    indicators,
    Elem(dict_prob_indicators[indicators[0]], Variable(indicators[0])),
)


# %%
# Choice model
# Read the estimates from the sequential estimation, and use
# them as starting values
MODELNAME = 'm02_sequential_estimation'
try:
    choice_results = res.bioResults(pickleFile=f'saved_results/{MODELNAME}.pickle')
except excep.BiogemeError:
    print(
        f'Run first the script {MODELNAME}.py in order to generate the '
        f'file {MODELNAME}.pickle, and move it to the directory saved_results'
    )
    sys.exit()
choice_betas = choice_results.getBetaValues()

# %%
ASC_CAR = Beta('ASC_CAR', choice_betas['ASC_CAR'], None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', choice_betas['ASC_SM'], None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', choice_betas['BETA_COST_HWH'], None, None, 0)
BETA_COST_OTHER = Beta(
    'BETA_COST_OTHER', choice_betas['BETA_COST_OTHER'], None, None, 0
)
BETA_DIST = Beta('BETA_DIST', choice_betas['BETA_DIST'], None, None, 0)
BETA_TIME_CAR_REF = Beta(
    'BETA_TIME_CAR_REF', choice_betas['BETA_TIME_CAR_REF'], None, 0, 0
)
BETA_TIME_CAR_AL = Beta(
    'BETA_TIME_CAR_AL', choice_betas['BETA_TIME_CAR_AL'], None, None, 0
)
BETA_TIME_PT_REF = Beta(
    'BETA_TIME_PT_REF', choice_betas['BETA_TIME_PT_REF'], None, 0, 0
)
BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_AL * ACTIVELIFE)
BETA_TIME_PT_AL = Beta(
    'BETA_TIME_PT_AL', choice_betas['BETA_TIME_PT_AL'], None, None, 0
)
BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_AL * ACTIVELIFE)
BETA_WAITING_TIME = Beta(
    'BETA_WAITING_TIME', choice_betas['BETA_WAITING_TIME'], None, None, 0
)

# %%
# Definition of utility functions:
V0 = (
    ASC_PT
    + BETA_TIME_PT * TimePT_scaled
    + BETA_WAITING_TIME * WaitingTimePT
    + BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH
    + BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
)

V1 = (
    ASC_CAR
    + BETA_TIME_CAR * TimeCar_scaled
    + BETA_COST_HWH * CostCarCHF_scaled * PurpHWH
    + BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
)

V2 = ASC_SM + BETA_DIST * distance_km_scaled

# %%
# Associate utility functions with the numbering of alternatives
V = {0: V0, 1: V1, 2: V2}

# %%
# Conditional on omega, we have a logit model (called the kernel) for
# the choice
condprob = models.logit(V, None, Choice)

# %%
# Conditional on omega, we have the product of ordered probit for the
# indicators.
condlike = prob_indicators * condprob

# %%
# We integrate over omega using numerical integration
loglike = log(Integrate(condlike * density, 'omega'))

# %%
# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'm03_simultaneous_estimation'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results = read_or_estimate(the_biogeme=the_biogeme, directory='saved_results')

# %%
print(results.short_summary())

# %%
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')

# %%
results.getEstimatedParameters()
