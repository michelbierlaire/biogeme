"""File b05latent_choice_full_mc.py

Choice model with the latent variable.
Mixture of logit.
Measurement equation for the indicators.
Maximum likelihood (full information) estimation.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 18:11:54 2023
"""

import sys
import biogeme.logging as blog
import biogeme.biogeme as bio
import biogeme.exceptions as excep
from biogeme import models
import biogeme.results as res
from biogeme.expressions import (
    Beta,
    log,
    bioDraws,
    MonteCarlo,
    Elem,
    bioNormalCdf,
    exp,
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
    Envir01,
    Envir02,
    Envir03,
    Mobil11,
    Mobil14,
    Mobil16,
    Mobil17,
    Choice,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b05latent_choice_full_mc.py')


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

coef_intercept = Beta('coef_intercept', structBetas['coef_intercept'], None, None, 0)
coef_age_65_more = Beta(
    'coef_age_65_more', structBetas['coef_age_65_more'], None, None, 0
)
coef_haveGA = Beta('coef_haveGA', structBetas['coef_haveGA'], None, None, 0)

coef_moreThanOneCar = Beta(
    'coef_moreThanOneCar', structBetas['coef_moreThanOneCar'], None, None, 0
)
coef_moreThanOneBike = Beta(
    'coef_moreThanOneBike', structBetas['coef_moreThanOneBike'], None, None, 0
)
coef_individualHouse = Beta(
    'coef_individualHouse', structBetas['coef_individualHouse'], None, None, 0
)
coef_male = Beta('coef_male', structBetas['coef_male'], None, None, 0)
coef_haveChildren = Beta(
    'coef_haveChildren', structBetas['coef_haveChildren'], None, None, 0
)
coef_highEducation = Beta(
    'coef_highEducation', structBetas['coef_highEducation'], None, None, 0
)

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

# Measurement equations

INTER_Envir01 = Beta('INTER_Envir01', 0, None, None, 1)
INTER_Envir02 = Beta('INTER_Envir02', 0, None, None, 0)
INTER_Envir03 = Beta('INTER_Envir03', 0, None, None, 0)
INTER_Mobil11 = Beta('INTER_Mobil11', 0, None, None, 0)
INTER_Mobil14 = Beta('INTER_Mobil14', 0, None, None, 0)
INTER_Mobil16 = Beta('INTER_Mobil16', 0, None, None, 0)
INTER_Mobil17 = Beta('INTER_Mobil17', 0, None, None, 0)

B_Envir01_F1 = Beta('B_Envir01_F1', -1, None, None, 1)
B_Envir02_F1 = Beta('B_Envir02_F1', -1, None, None, 0)
B_Envir03_F1 = Beta('B_Envir03_F1', 1, None, None, 0)
B_Mobil11_F1 = Beta('B_Mobil11_F1', 1, None, None, 0)
B_Mobil14_F1 = Beta('B_Mobil14_F1', 1, None, None, 0)
B_Mobil16_F1 = Beta('B_Mobil16_F1', 1, None, None, 0)
B_Mobil17_F1 = Beta('B_Mobil17_F1', 1, None, None, 0)

MODEL_Envir01 = INTER_Envir01 + B_Envir01_F1 * CARLOVERS
MODEL_Envir02 = INTER_Envir02 + B_Envir02_F1 * CARLOVERS
MODEL_Envir03 = INTER_Envir03 + B_Envir03_F1 * CARLOVERS
MODEL_Mobil11 = INTER_Mobil11 + B_Mobil11_F1 * CARLOVERS
MODEL_Mobil14 = INTER_Mobil14 + B_Mobil14_F1 * CARLOVERS
MODEL_Mobil16 = INTER_Mobil16 + B_Mobil16_F1 * CARLOVERS
MODEL_Mobil17 = INTER_Mobil17 + B_Mobil17_F1 * CARLOVERS

SIGMA_STAR_Envir01 = Beta('SIGMA_STAR_Envir01', 1, 1.0e-5, None, 1)
SIGMA_STAR_Envir02 = Beta('SIGMA_STAR_Envir02', 1, 1.0e-5, None, 0)
SIGMA_STAR_Envir03 = Beta('SIGMA_STAR_Envir03', 1, 1.0e-5, None, 0)
SIGMA_STAR_Mobil11 = Beta('SIGMA_STAR_Mobil11', 1, 1.0e-5, None, 0)
SIGMA_STAR_Mobil14 = Beta('SIGMA_STAR_Mobil14', 1, 1.0e-5, None, 0)
SIGMA_STAR_Mobil16 = Beta('SIGMA_STAR_Mobil16', 1, 1.0e-5, None, 0)
SIGMA_STAR_Mobil17 = Beta('SIGMA_STAR_Mobil17', 1, 1.0e-5, None, 0)


delta_1 = Beta('delta_1', 0.1, 1.0e-5, None, 0)
delta_2 = Beta('delta_2', 0.2, 1.0e-5, None, 0)
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1
tau_3 = delta_1
tau_4 = delta_1 + delta_2

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

# Choice model
# Read the estimates from the sequential estimation, and use
# them as starting values

FILENAME = 'b04latent_choice_seq'
try:
    choiceResults = res.bioResults(pickleFile=f'{FILENAME}.pickle')
except excep.BiogemeError:
    print(
        f'Run first the script {FILENAME}.py in order to generate the '
        f'file {FILENAME}.pickle.'
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
BETA_TIME_CAR_CL = Beta(
    'BETA_TIME_CAR_CL', choiceBetas['BETA_TIME_CAR_CL'], None, None, 0
)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', choiceBetas['BETA_TIME_PT_REF'], None, 0, 0)
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL', choiceBetas['BETA_TIME_PT_CL'], None, None, 0)
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

# Conditional to omega, we have a logit model (called the kernel) for
# the choice
condprob = models.logit(V, None, Choice)

# Conditional to omega, we have the product of ordered probit for the
# indicators.
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

# We integrate over omega using numerical integration
loglike = log(MonteCarlo(condlike))

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'b05latent_choice_full_mc'
# Estimate the parameters
results = the_biogeme.estimate()


print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'Final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
results.writeLaTeX()
print(f'LaTeX file: {results.data.latexFileName}')
