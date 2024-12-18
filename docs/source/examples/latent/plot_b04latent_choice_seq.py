"""

Choice model with a latent variable: sequential estimation
==========================================================

Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 17:32:34 2023

"""

import sys

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
import biogeme.distributions as dist
from biogeme.biogeme import BIOGEME
from biogeme.data.optima import (
    read_data,
    age_65_more,
    moreThanOneCar,
    moreThanOneBike,
    individualHouse,
    male,
    haveChildren,
    haveGA,
    highEducation,
    WaitingTimePT,
    Choice,
    TimePT_scaled,
    TimeCar_scaled,
    MarginalCostPT_scaled,
    CostCarCHF_scaled,
    distance_km_scaled,
    PurpHWH,
    PurpOther,
    ScaledIncome,
)
from biogeme.expressions import (
    Beta,
    RandomVariable,
    exp,
    log,
    Integrate,
)
from biogeme.models import piecewise_formula, logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)
from read_or_estimate import read_or_estimate

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b04latent_choice_seq.py')


# %%
# Read the estimates from the structural equation estimation.
MODELNAME = 'b02one_latent_ordered'
try:
    struct_results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{MODELNAME}.yaml'
    )
except FileNotFoundError:
    print(
        f'Run first the script {MODELNAME}.py in order to generate the '
        f'file {MODELNAME}.yaml, and move it to the directory saved_results'
    )
    sys.exit()
struct_betas = struct_results.get_beta_values()

# %%
# Coefficients.
coef_intercept = struct_betas['coef_intercept']
coef_age_65_more = struct_betas['coef_age_65_more']
coef_haveGA = struct_betas['coef_haveGA']
coef_moreThanOneCar = struct_betas['coef_moreThanOneCar']
coef_moreThanOneBike = struct_betas['coef_moreThanOneBike']
coef_individualHouse = struct_betas['coef_individualHouse']
coef_male = struct_betas['coef_male']
coef_haveChildren = struct_betas['coef_haveChildren']
coef_highEducation = struct_betas['coef_highEducation']

# %%
# Latent variable: structural equation.

# %%
# Define a random parameter, normally distributed, designed to be used
# for numerical integration

omega = RandomVariable('omega')
density = dist.normalpdf(omega)
sigma_s = Beta('sigma_s', 1, None, None, 0)

thresholds = [None, 4, 6, 8, 10, None]
formula_income = piecewise_formula(
    variable=ScaledIncome,
    thresholds=thresholds,
    betas=[
        struct_betas['beta_ScaledIncome_minus_inf_4'],
        struct_betas['beta_ScaledIncome_4_6'],
        struct_betas['beta_ScaledIncome_6_8'],
        struct_betas['beta_ScaledIncome_8_10'],
        struct_betas['beta_ScaledIncome_10_inf'],
    ],
)


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
# Choice model.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0, None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', 0, None, None, 0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', 0, None, None, 0)
BETA_DIST = Beta('BETA_DIST', 0, None, None, 0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF', 0, None, 0, 0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', 0, None, 0, 0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', 0, None, None, 0)

# %%
# The coefficient of the latent variable should be initialized to
# something different from zero. If not, the algorithm may be trapped
# in a local optimum, and never change the value.
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL', -0.01, None, None, 0)
BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_CL * CARLOVERS)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL', -0.01, None, None, 0)
BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_CL * CARLOVERS)

# %%
# Definition of utility functions:.

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
# Associate utility functions with the numbering of alternatives.
V = {0: V0, 1: V1, 2: V2}

# %%
# Conditional on omega, we have a logit model (called the kernel).
condprob = logit(V, None, Choice)

# %%
# We integrate over omega using numerical integration.
loglike = log(Integrate(condprob * density, 'omega'))

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, loglike)
the_biogeme.modelName = 'b04latent_choice_seq'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results = read_or_estimate(the_biogeme=the_biogeme, directory='saved_results')

# %%
print(f'Estimated betas: {results.number_of_parameters}')
print(f'Final log likelihood: {results.final_log_likelihood:.3f}')
print(f'Output file: {the_biogeme.html_filename}')

# %%
# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
