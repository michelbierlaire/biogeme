"""

Mixture of logit
================

Choice model with latent_old variable. No measurement equation for the indicators.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 16:58:21 2023

"""

from optima import (
    Choice,
    CostCarCHF_scaled,
    MarginalCostPT_scaled,
    PurpHWH,
    PurpOther,
    ScaledIncome,
    TimeCar_scaled,
    TimePT_scaled,
    WaitingTimePT,
    age_65_more,
    database,
    distance_km_scaled,
    haveChildren,
    haveGA,
    highEducation,
    individualHouse,
    male,
    moreThanOneBike,
    moreThanOneCar,
)
from read_or_estimate import read_or_estimate

import biogeme.biogeme as bio
import biogeme.biogeme_logging as blog
import biogeme.distributions as dist
from biogeme import models
from biogeme.expressions import (
    Beta,
    Integrate,
    RandomVariable,
    exp,
    log,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b03choice_only.py')

# %%
# Parameters to be estimated
coef_intercept = Beta('coef_intercept', 0.0, None, None, 1)
coef_age_65_more = Beta('coef_age_65_more', 0.0, None, None, 0)
coef_haveGA = Beta('coef_haveGA', 0.0, None, None, 0)
coef_moreThanOneCar = Beta('coef_moreThanOneCar', 0.0, None, None, 0)
coef_moreThanOneBike = Beta('coef_moreThanOneBike', 0.0, None, None, 0)
coef_individualHouse = Beta('coef_individualHouse', 0.0, None, None, 0)
coef_male = Beta('coef_male', 0.0, None, None, 0)
coef_haveChildren = Beta('coef_haveChildren', 0.0, None, None, 0)
coef_highEducation = Beta('coef_highEducation', 0.0, None, None, 0)

# %%
# Latent variable: structural equation.

# %%
# Define a random parameter, normally distributed)
# designed to be used
# for numerical integration.
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s = Beta('sigma_s', 1, None, None, 0)

thresholds = [None, 4, 6, 8, 10, None]
formula_income = models.piecewiseFormula(variable=ScaledIncome, thresholds=thresholds)

CARLOVERS = (
    coef_intercept
    + coef_age_65_more * age_65_more
    + formula_income
    + coef_moreThanOneCar * moreThanOneCar
    + coef_moreThanOneBike * moreThanOneBike
    + coef_individualHouse * individualHouse
    + coef_male * male
    + coef_haveChildren * haveChildren
    + coef_haveGA * haveGA
    + coef_highEducation * highEducation
    + sigma_s * omega
)

# %%
# Choice model: parameters.
ASC_CAR = Beta('ASC_CAR', 0.0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0.0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0.0, None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', 0.0, None, None, 0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', 0.0, None, None, 0)
BETA_DIST = Beta('BETA_DIST', 0.0, None, None, 0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF', -0.0001, None, 0, 0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL', -1.0, -3, 3, 0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', -0.0001, None, 0, 0)
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL', -1.0, -3, 3, 0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', 0.0, None, None, 0)

# %%
# Definition of utility functions.
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

# %%
# Associate utility functions with the numbering of alternatives.
V = {0: V0, 1: V1, 2: V2}

# %%
# Conditional on  omega, we have a logit model (called the kernel).
condprob = models.logit(V, None, Choice)

# %%
# We integrate over omega using numerical integration.
loglike = log(Integrate(condprob * density, 'omega'))


# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'b03choice_only'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results = read_or_estimate(the_biogeme=the_biogeme, directory='saved_results')

# %%
print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')

# %%
results.getEstimatedParameters()
