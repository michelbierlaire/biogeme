"""

Serial correlation
==================

Choice model with the latent_old variable.
Mixture of logit, with agent effect to deal with serial correlation.
Measurement equation for the indicators.
Maximum likelihood (full information) estimation.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 18:16:37 2023

"""

import sys

from optima import (
    Choice,
    CostCarCHF_scaled,
    Envir01,
    Envir02,
    Envir03,
    MarginalCostPT_scaled,
    Mobil11,
    Mobil14,
    Mobil16,
    Mobil17,
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
import biogeme.exceptions as excep
import biogeme.results as res
from biogeme import models
from biogeme.expressions import (
    Beta,
    Elem,
    MonteCarlo,
    bioDraws,
    bioNormalCdf,
    exp,
    log,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b06serial_correlation.py')

# %%
# Read the estimates from the structural equation estimation.
MODELNAME = 'b05latent_choice_full'
try:
    struct_results = res.bioResults(pickleFile=f'saved_results/{MODELNAME}.pickle')
except excep.BiogemeError:
    print(
        f'Run first the script {MODELNAME}.py in order to generate the '
        f'file {MODELNAME}.pickle, and move it to the directory saved_results'
    )
    sys.exit()
betas = struct_results.getBetaValues()

# %%
# Coefficients.
coef_intercept = Beta('coef_intercept', betas['coef_intercept'], None, None, 0)
coef_age_65_more = Beta('coef_age_65_more', betas['coef_age_65_more'], None, None, 0)
coef_haveGA = Beta('coef_haveGA', betas['coef_haveGA'], None, None, 0)

coef_moreThanOneCar = Beta(
    'coef_moreThanOneCar', betas['coef_moreThanOneCar'], None, None, 0
)
coef_moreThanOneBike = Beta(
    'coef_moreThanOneBike', betas['coef_moreThanOneBike'], None, None, 0
)
coef_individualHouse = Beta(
    'coef_individualHouse', betas['coef_individualHouse'], None, None, 0
)
coef_male = Beta('coef_male', betas['coef_male'], None, None, 0)
coef_haveChildren = Beta('coef_haveChildren', betas['coef_haveChildren'], None, None, 0)
coef_highEducation = Beta(
    'coef_highEducation', betas['coef_highEducation'], None, None, 0
)

# %%
# Latent variable: structural equation.

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo integration.
omega = bioDraws('omega', 'NORMAL')
sigma_s = Beta('sigma_s', betas['sigma_s'], None, None, 0)

# %%
# Deal with serial correlation by including an error component
# that is individual specific
error_component = bioDraws('error_component', 'NORMAL')
ec_sigma = Beta('ec_sigma', 10, None, None, 0)

thresholds = [None, 4, 6, 8, 10, None]
betas_thresholds = [
    Beta(
        'beta_ScaledIncome_minus_inf_4',
        betas['beta_ScaledIncome_minus_inf_4'],
        None,
        None,
        0,
    ),
    Beta(
        'beta_ScaledIncome_4_6',
        betas['beta_ScaledIncome_4_6'],
        None,
        None,
        0,
    ),
    Beta(
        'beta_ScaledIncome_6_8',
        betas['beta_ScaledIncome_6_8'],
        None,
        None,
        0,
    ),
    Beta(
        'beta_ScaledIncome_8_10',
        betas['beta_ScaledIncome_8_10'],
        None,
        None,
        0,
    ),
    Beta(
        'beta_ScaledIncome_10_inf',
        betas['beta_ScaledIncome_10_inf'],
        None,
        None,
        0,
    ),
]

formula_income = models.piecewiseFormula(
    variable=ScaledIncome,
    thresholds=thresholds,
    betas=betas_thresholds,
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
    + ec_sigma * error_component
)

# %%
# Measurement equations.

# %%
# Intercepts.
INTER_Envir01 = Beta('INTER_Envir01', 0, None, None, 1)
INTER_Envir02 = Beta('INTER_Envir02', betas['INTER_Envir02'], None, None, 0)
INTER_Envir03 = Beta('INTER_Envir03', betas['INTER_Envir03'], None, None, 0)
INTER_Mobil11 = Beta('INTER_Mobil11', betas['INTER_Mobil11'], None, None, 0)
INTER_Mobil14 = Beta('INTER_Mobil14', betas['INTER_Mobil14'], None, None, 0)
INTER_Mobil16 = Beta('INTER_Mobil16', betas['INTER_Mobil16'], None, None, 0)
INTER_Mobil17 = Beta('INTER_Mobil17', betas['INTER_Mobil17'], None, None, 0)

# %%
# Coefficients.
B_Envir01_F1 = Beta('B_Envir01_F1', -1, None, None, 1)
B_Envir02_F1 = Beta('B_Envir02_F1', betas['B_Envir02_F1'], None, None, 0)
B_Envir03_F1 = Beta('B_Envir03_F1', betas['B_Envir03_F1'], None, None, 0)
B_Mobil11_F1 = Beta('B_Mobil11_F1', betas['B_Mobil11_F1'], None, None, 0)
B_Mobil14_F1 = Beta('B_Mobil14_F1', betas['B_Mobil14_F1'], None, None, 0)
B_Mobil16_F1 = Beta('B_Mobil16_F1', betas['B_Mobil16_F1'], None, None, 0)
B_Mobil17_F1 = Beta('B_Mobil17_F1', betas['B_Mobil17_F1'], None, None, 0)

# %%
# Linear models.
MODEL_Envir01 = INTER_Envir01 + B_Envir01_F1 * CARLOVERS
MODEL_Envir02 = INTER_Envir02 + B_Envir02_F1 * CARLOVERS
MODEL_Envir03 = INTER_Envir03 + B_Envir03_F1 * CARLOVERS
MODEL_Mobil11 = INTER_Mobil11 + B_Mobil11_F1 * CARLOVERS
MODEL_Mobil14 = INTER_Mobil14 + B_Mobil14_F1 * CARLOVERS
MODEL_Mobil16 = INTER_Mobil16 + B_Mobil16_F1 * CARLOVERS
MODEL_Mobil17 = INTER_Mobil17 + B_Mobil17_F1 * CARLOVERS

# %%
# Scale parameters.
SIGMA_STAR_Envir01 = Beta('SIGMA_STAR_Envir01', 1, None, None, 1)
SIGMA_STAR_Envir02 = Beta(
    'SIGMA_STAR_Envir02', betas['SIGMA_STAR_Envir02'], None, None, 0
)
SIGMA_STAR_Envir03 = Beta(
    'SIGMA_STAR_Envir03', betas['SIGMA_STAR_Envir03'], None, None, 0
)
SIGMA_STAR_Mobil11 = Beta(
    'SIGMA_STAR_Mobil11', betas['SIGMA_STAR_Mobil11'], None, None, 0
)
SIGMA_STAR_Mobil14 = Beta(
    'SIGMA_STAR_Mobil14', betas['SIGMA_STAR_Mobil14'], None, None, 0
)
SIGMA_STAR_Mobil16 = Beta(
    'SIGMA_STAR_Mobil16', betas['SIGMA_STAR_Mobil16'], None, None, 0
)
SIGMA_STAR_Mobil17 = Beta(
    'SIGMA_STAR_Mobil17', betas['SIGMA_STAR_Mobil17'], None, None, 0
)

# %%
# Symmetric thresholds.
delta_1 = Beta('delta_1', betas['delta_1'], 0, 10, 0)
delta_2 = Beta('delta_2', betas['delta_2'], 0, 10, 0)
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1
tau_3 = delta_1
tau_4 = delta_1 + delta_2

# %%
# Ordered probit models.
Envir01_tau_1 = (tau_1 - MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_2 = (tau_2 - MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_3 = (tau_3 - MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_4 = (tau_4 - MODEL_Envir01) / SIGMA_STAR_Envir01
IndEnvir01 = {
    1: bioNormalCdf(Envir01_tau_1),
    2: bioNormalCdf(Envir01_tau_2) - bioNormalCdf(Envir01_tau_1),
    3: bioNormalCdf(Envir01_tau_3) - bioNormalCdf(Envir01_tau_2),
    4: bioNormalCdf(Envir01_tau_4) - bioNormalCdf(Envir01_tau_3),
    5: 1 - bioNormalCdf(Envir01_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_Envir01 = Elem(IndEnvir01, Envir01)


Envir02_tau_1 = (tau_1 - MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_2 = (tau_2 - MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_3 = (tau_3 - MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_4 = (tau_4 - MODEL_Envir02) / SIGMA_STAR_Envir02
IndEnvir02 = {
    1: bioNormalCdf(Envir02_tau_1),
    2: bioNormalCdf(Envir02_tau_2) - bioNormalCdf(Envir02_tau_1),
    3: bioNormalCdf(Envir02_tau_3) - bioNormalCdf(Envir02_tau_2),
    4: bioNormalCdf(Envir02_tau_4) - bioNormalCdf(Envir02_tau_3),
    5: 1 - bioNormalCdf(Envir02_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_Envir02 = Elem(IndEnvir02, Envir02)

Envir03_tau_1 = (tau_1 - MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_2 = (tau_2 - MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_3 = (tau_3 - MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_4 = (tau_4 - MODEL_Envir03) / SIGMA_STAR_Envir03
IndEnvir03 = {
    1: bioNormalCdf(Envir03_tau_1),
    2: bioNormalCdf(Envir03_tau_2) - bioNormalCdf(Envir03_tau_1),
    3: bioNormalCdf(Envir03_tau_3) - bioNormalCdf(Envir03_tau_2),
    4: bioNormalCdf(Envir03_tau_4) - bioNormalCdf(Envir03_tau_3),
    5: 1 - bioNormalCdf(Envir03_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_Envir03 = Elem(IndEnvir03, Envir03)

Mobil11_tau_1 = (tau_1 - MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_2 = (tau_2 - MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_3 = (tau_3 - MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_4 = (tau_4 - MODEL_Mobil11) / SIGMA_STAR_Mobil11
IndMobil11 = {
    1: bioNormalCdf(Mobil11_tau_1),
    2: bioNormalCdf(Mobil11_tau_2) - bioNormalCdf(Mobil11_tau_1),
    3: bioNormalCdf(Mobil11_tau_3) - bioNormalCdf(Mobil11_tau_2),
    4: bioNormalCdf(Mobil11_tau_4) - bioNormalCdf(Mobil11_tau_3),
    5: 1 - bioNormalCdf(Mobil11_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_Mobil11 = Elem(IndMobil11, Mobil11)

Mobil14_tau_1 = (tau_1 - MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_2 = (tau_2 - MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_3 = (tau_3 - MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_4 = (tau_4 - MODEL_Mobil14) / SIGMA_STAR_Mobil14
IndMobil14 = {
    1: bioNormalCdf(Mobil14_tau_1),
    2: bioNormalCdf(Mobil14_tau_2) - bioNormalCdf(Mobil14_tau_1),
    3: bioNormalCdf(Mobil14_tau_3) - bioNormalCdf(Mobil14_tau_2),
    4: bioNormalCdf(Mobil14_tau_4) - bioNormalCdf(Mobil14_tau_3),
    5: 1 - bioNormalCdf(Mobil14_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_Mobil14 = Elem(IndMobil14, Mobil14)

Mobil16_tau_1 = (tau_1 - MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_2 = (tau_2 - MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_3 = (tau_3 - MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_4 = (tau_4 - MODEL_Mobil16) / SIGMA_STAR_Mobil16
IndMobil16 = {
    1: bioNormalCdf(Mobil16_tau_1),
    2: bioNormalCdf(Mobil16_tau_2) - bioNormalCdf(Mobil16_tau_1),
    3: bioNormalCdf(Mobil16_tau_3) - bioNormalCdf(Mobil16_tau_2),
    4: bioNormalCdf(Mobil16_tau_4) - bioNormalCdf(Mobil16_tau_3),
    5: 1 - bioNormalCdf(Mobil16_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_Mobil16 = Elem(IndMobil16, Mobil16)

Mobil17_tau_1 = (tau_1 - MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_2 = (tau_2 - MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_3 = (tau_3 - MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_4 = (tau_4 - MODEL_Mobil17) / SIGMA_STAR_Mobil17
IndMobil17 = {
    1: bioNormalCdf(Mobil17_tau_1),
    2: bioNormalCdf(Mobil17_tau_2) - bioNormalCdf(Mobil17_tau_1),
    3: bioNormalCdf(Mobil17_tau_3) - bioNormalCdf(Mobil17_tau_2),
    4: bioNormalCdf(Mobil17_tau_4) - bioNormalCdf(Mobil17_tau_3),
    5: 1 - bioNormalCdf(Mobil17_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0,
}

P_Mobil17 = Elem(IndMobil17, Mobil17)

# %%
# Choice model.
ASC_CAR = Beta('ASC_CAR', betas['ASC_CAR'], None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', betas['ASC_SM'], None, None, 0)
BETA_COST_HWH = Beta('BETA_COST_HWH', betas['BETA_COST_HWH'], None, None, 0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', betas['BETA_COST_OTHER'], None, None, 0)
BETA_DIST = Beta('BETA_DIST', betas['BETA_DIST'], None, None, 0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF', betas['BETA_TIME_CAR_REF'], None, 0, 0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL', betas['BETA_TIME_CAR_CL'], -10, 10, 0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', betas['BETA_TIME_PT_REF'], None, 0, 0)
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL', betas['BETA_TIME_PT_CL'], -10, 10, 0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', betas['BETA_WAITING_TIME'], None, None, 0)

# %%
# Definition of utility functions.
BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_CL * CARLOVERS)

V0 = (
    ASC_PT
    + BETA_TIME_PT * TimePT_scaled
    + BETA_WAITING_TIME * WaitingTimePT
    + BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH
    + BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther
    + ec_sigma * error_component
)

BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_CL * CARLOVERS)

V1 = (
    ASC_CAR
    + BETA_TIME_CAR * TimeCar_scaled
    + BETA_COST_HWH * CostCarCHF_scaled * PurpHWH
    + BETA_COST_OTHER * CostCarCHF_scaled * PurpOther
    + ec_sigma * error_component
)

V2 = ASC_SM + BETA_DIST * distance_km_scaled

# %%
# Associate utility functions with the numbering of alternatives.
V = {0: V0, 1: V1, 2: V2}

# %%
# Conditional on the random parameters, we have a logit model (called
# the kernel) for the choice.
condprob = models.logit(V, None, Choice)

# %%
# Conditional on the random parameters, we have the product of ordered
# probit for the indicators.
condlike = (
    P_Envir01
    * P_Envir02
    * P_Envir03
    * P_Mobil11
    * P_Mobil14
    * P_Mobil16
    * P_Mobil17
    * condprob
)

# %%
# We integrate over omega using Monte-Carlo integration
loglike = log(MonteCarlo(condlike))

# %%
# Create the Biogeme object. As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws. To achieve that, we provide a parameter file
# different from the default one.
the_biogeme = bio.BIOGEME(database, loglike, parameter_file='few_draws.toml')
the_biogeme.modelName = 'b06serial_correlation'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results = read_or_estimate(the_biogeme=the_biogeme, directory='saved_results')

# %%
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')

# %%
results.getEstimatedParameters()
