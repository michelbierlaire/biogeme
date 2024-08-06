"""

Measurement equations: continuous indicators
============================================

It is actually a simle linear regression.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 16:42:02 2023
"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.models import piecewiseFormula
import biogeme.loglikelihood as ll
from biogeme.expressions import Beta, Elem, bioMultSum
from optima import (
    database,
    age_65_more,
    ScaledIncome,
    moreThanOneCar,
    moreThanOneBike,
    individualHouse,
    male,
    haveChildren,
    haveGA,
    highEducation,
    Envir01,
    Envir02,
    Envir03,
    Mobil11,
    Mobil14,
    Mobil16,
    Mobil17,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01one_latent_regression.py')

# %%
# Parameters to be estimated.
coef_intercept = Beta('coef_intercept', 0.0, None, None, 0)
coef_age_65_more = Beta('coef_age_65_more', 0.0, None, None, 0)
coef_haveGA = Beta('coef_haveGA', 0.0, None, None, 0)
coef_moreThanOneCar = Beta('coef_moreThanOneCar', 0.0, None, None, 0)
coef_moreThanOneBike = Beta('coef_moreThanOneBike', 0.0, None, None, 0)
coef_individualHouse = Beta('coef_individualHouse', 0.0, None, None, 0)
coef_male = Beta('coef_male', 0.0, None, None, 0)
coef_haveChildren = Beta('coef_haveChildren', 0.0, None, None, 0)
coef_highEducation = Beta('coef_highEducation', 0.0, None, None, 0)

thresholds = [None, 4, 6, 8, 10, None]
formula_income = piecewiseFormula(variable=ScaledIncome, thresholds=thresholds)

# %%
# Latent variable: structural equation.
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
)


# %%
# Measurement equations.

# %%
# Intercepts.
INTER_Envir01 = Beta('INTER_Envir01', 0, None, None, 1)
INTER_Envir02 = Beta('INTER_Envir02', 0, None, None, 0)
INTER_Envir03 = Beta('INTER_Envir03', 0, None, None, 0)
INTER_Mobil11 = Beta('INTER_Mobil11', 0, None, None, 0)
INTER_Mobil14 = Beta('INTER_Mobil14', 0, None, None, 0)
INTER_Mobil16 = Beta('INTER_Mobil16', 0, None, None, 0)
INTER_Mobil17 = Beta('INTER_Mobil17', 0, None, None, 0)

# %%
# Coefficients.
B_Envir01_F1 = Beta('B_Envir01_F1', -1, None, None, 1)
B_Envir02_F1 = Beta('B_Envir02_F1', -1, None, None, 0)
B_Envir03_F1 = Beta('B_Envir03_F1', 1, None, None, 0)
B_Mobil11_F1 = Beta('B_Mobil11_F1', 1, None, None, 0)
B_Mobil14_F1 = Beta('B_Mobil14_F1', 1, None, None, 0)
B_Mobil16_F1 = Beta('B_Mobil16_F1', 1, None, None, 0)
B_Mobil17_F1 = Beta('B_Mobil17_F1', 1, None, None, 0)

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
SIGMA_STAR_Envir01 = Beta('SIGMA_STAR_Envir01', 1, None, None, 0)
SIGMA_STAR_Envir02 = Beta('SIGMA_STAR_Envir02', 1, None, None, 0)
SIGMA_STAR_Envir03 = Beta('SIGMA_STAR_Envir03', 1, None, None, 0)
SIGMA_STAR_Mobil11 = Beta('SIGMA_STAR_Mobil11', 1, None, None, 0)
SIGMA_STAR_Mobil14 = Beta('SIGMA_STAR_Mobil14', 1, None, None, 0)
SIGMA_STAR_Mobil16 = Beta('SIGMA_STAR_Mobil16', 1, None, None, 0)
SIGMA_STAR_Mobil17 = Beta('SIGMA_STAR_Mobil17', 1, None, None, 0)

# %%
# We build a dict with each contribution to the loglikelihood if
# (var > 0) and (var < 6). If not, 0 is returned.
F = {}
F['Envir01'] = Elem(
    {
        0: 0,
        1: ll.loglikelihoodregression(Envir01, MODEL_Envir01, SIGMA_STAR_Envir01),
    },
    (Envir01 > 0) * (Envir01 < 6),
)
F['Envir02'] = Elem(
    {
        0: 0,
        1: ll.loglikelihoodregression(Envir02, MODEL_Envir02, SIGMA_STAR_Envir02),
    },
    (Envir02 > 0) * (Envir02 < 6),
)
F['Envir03'] = Elem(
    {
        0: 0,
        1: ll.loglikelihoodregression(Envir03, MODEL_Envir03, SIGMA_STAR_Envir03),
    },
    (Envir03 > 0) * (Envir03 < 6),
)
F['Mobil11'] = Elem(
    {
        0: 0,
        1: ll.loglikelihoodregression(Mobil11, MODEL_Mobil11, SIGMA_STAR_Mobil11),
    },
    (Mobil11 > 0) * (Mobil11 < 6),
)
F['Mobil14'] = Elem(
    {
        0: 0,
        1: ll.loglikelihoodregression(Mobil14, MODEL_Mobil14, SIGMA_STAR_Mobil14),
    },
    (Mobil14 > 0) * (Mobil14 < 6),
)
F['Mobil16'] = Elem(
    {
        0: 0,
        1: ll.loglikelihoodregression(Mobil16, MODEL_Mobil16, SIGMA_STAR_Mobil16),
    },
    (Mobil16 > 0) * (Mobil16 < 6),
)
F['Mobil17'] = Elem(
    {
        0: 0,
        1: ll.loglikelihoodregression(Mobil17, MODEL_Mobil17, SIGMA_STAR_Mobil17),
    },
    (Mobil17 > 0) * (Mobil17 < 6),
)

# %%
# The log likelihood is the sum of the elements of the above dict
loglike = bioMultSum(F)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, loglike)
the_biogeme.modelName = 'b01one_latent_regression'

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(f'Estimated betas: {len(results.data.betaValues)}')
print(f'final log likelihood: {results.data.logLike:.3f}')
print(f'Output file: {results.data.htmlFileName}')
results.writeLaTeX()
print(f'LaTeX file: {results.data.latexFileName}')

# %%
results.getEstimatedParameters()
