"""

Choice model with latent_old variable: sequential estimation
========================================================

Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 09:47:53 2023

"""

import sys

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
    Integrate,
    RandomVariable,
    exp,
    log,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example m02_sequential_estimation.py')

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
coef_intercept = struct_betas['coef_intercept']
coef_age_30_less = struct_betas['coef_age_30_less']
coef_male = struct_betas['coef_male']
coef_haveChildren = struct_betas['coef_haveChildren']
coef_highEducation = struct_betas['coef_highEducation']
coef_artisans = struct_betas['coef_artisans']
coef_employees = struct_betas['coef_employees']
coef_child_center = struct_betas['coef_child_center']
coef_child_suburb = struct_betas['coef_child_suburb']

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
# Choice model
ASC_CAR = Beta('ASC_CAR', 0.94, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0.35, None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', -2.3, None, None, 0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', -1.9, None, None, 0)
BETA_DIST = Beta('BETA_DIST', -1.3, None, None, 0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF', -6.1, None, 0, 0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', 0, None, 0, 0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', -0.075, None, None, 0)

# %%
# The coefficient of the latent_old variable should be initialized to
# something different from zero. If not, the algorithm may be trapped
# in a local optimum, and never change the value.
BETA_TIME_PT_AL = Beta('BETA_TIME_PT_AL', 1.5, None, None, 0)
BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_AL * ACTIVELIFE)
BETA_TIME_CAR_AL = Beta('BETA_TIME_CAR_AL', -0.11, None, None, 0)
BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_AL * ACTIVELIFE)

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
# Conditional on omega, we have a logit model (called the kernel)
condprob = models.logit(V, None, Choice)

# %%
# We integrate over omega using numerical integration
loglike = log(Integrate(condprob * density, 'omega'))

# %%
# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'm02_sequential_estimation'
the_biogeme.maxiter = 1000

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
