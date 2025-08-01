"""File generalized_specification.py

Michel Bierlaire, EPFL
Fri Jul 25 2025, 16:36:06

Specification of the "generalized translated utility" MDCEV model.
"""

from biogeme.expressions import Beta
from biogeme.mdcev import Generalized
from specification import baseline_utilities, weight

# %
# Gamma parameters. Must be positive.
lowest_positive_value = 0.0001
gamma_shopping = Beta('gamma_shopping', 1, lowest_positive_value, None, 0)
gamma_socializing = Beta('gamma_socializing', 1, lowest_positive_value, None, 0)
gamma_recreation = Beta('gamma_recreation', 1, lowest_positive_value, None, 0)
gamma_personal = Beta('gamma_personal', 1, lowest_positive_value, None, 0)

# %
# alpha parameters. Must be between 0 and 1
alpha_shopping = Beta(
    'alpha_shopping', 0.5, lowest_positive_value, 1 - lowest_positive_value, 0
)
alpha_socializing = Beta(
    'alpha_socializing', 0.5, lowest_positive_value, 1 - lowest_positive_value, 0
)
alpha_recreation = Beta(
    'alpha_recreation', 0.5, lowest_positive_value, 1 - lowest_positive_value, 0
)
alpha_personal = Beta(
    'alpha_personal', 0.5, lowest_positive_value, 1 - lowest_positive_value, 0
)

scale_parameter = Beta('scale', 1, lowest_positive_value, None, 0)

gamma_parameters = {
    1: gamma_shopping,
    2: gamma_socializing,
    3: gamma_recreation,
    4: gamma_personal,
}

alpha_parameters = {
    1: alpha_shopping,
    2: alpha_socializing,
    3: alpha_recreation,
    4: alpha_personal,
}

the_generalized = Generalized(
    model_name='generalized',
    baseline_utilities=baseline_utilities,
    gamma_parameters=gamma_parameters,
    alpha_parameters=alpha_parameters,
    scale_parameter=scale_parameter,
    weights=weight,
)
