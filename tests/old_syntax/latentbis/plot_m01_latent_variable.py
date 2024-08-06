"""

Measurement equations: discrete indicators
==========================================

Ordered probit.

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 09:39:10 2023

"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme.expressions import (
    Beta,
    log,
    Elem,
    bioNormalCdf,
    Variable,
    bioMultSum,
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
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example m01_latent_variable.py')

# %%
# Parameters for the structural equation
coef_intercept = Beta('coef_intercept', 0.0, None, None, 0)
coef_age_30_less = Beta('coef_age_30_less', 0.0, None, None, 0)
coef_male = Beta('coef_male', 0.0, None, None, 0)
coef_haveChildren = Beta('coef_haveChildren', 0.0, None, None, 0)
coef_highEducation = Beta('coef_highEducation', 0.0, None, None, 0)
coef_artisans = Beta('coef_artisans', 0.0, None, None, 0)
coef_employees = Beta('coef_employees', 0.0, None, None, 0)
coef_child_center = Beta('coef_child_center', 0.0, None, None, 0)
coef_child_suburb = Beta('coef_child_suburb', 0.0, None, None, 0)

# %%
# Latent variable: structural equation
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
models = {k: inter[k] + coefficients[k] * ACTIVELIFE for k in indicators}

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
tau_1_residual = {k: (tau_1 - models[k]) / sigma_star[k] for k in indicators}
tau_2_residual = {k: (tau_2 - models[k]) / sigma_star[k] for k in indicators}
tau_3_residual = {k: (tau_3 - models[k]) / sigma_star[k] for k in indicators}
tau_4_residual = {k: (tau_4 - models[k]) / sigma_star[k] for k in indicators}
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
log_proba = {k: log(Elem(dict_prob_indicators[k], Variable(k))) for k in indicators}
loglike = bioMultSum(log_proba)

# %%
# Create the Biogeme object
biogeme = bio.BIOGEME(database, loglike)
biogeme.modelName = 'm01_latent_variable'

# %%
# Estimate the parameters
results = biogeme.estimate()

# %%
results.getEstimatedParameters()
