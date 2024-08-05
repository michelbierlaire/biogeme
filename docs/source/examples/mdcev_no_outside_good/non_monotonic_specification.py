"""File non_monotonic_specification.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 18:18:38 2024

Specification of the "non monotonic utility" MDCEV model.
"""

from biogeme.expressions import Beta
from biogeme.mdcev import NonMonotonic
from specification import (
    weight,
    baseline_utilities,
    mu_utilities,
    hhsize,
    employed,
    spousepr,
    male,
    age15_40,
    age41_60,
    bachigher,
    white,
    metro,
    Sunday,
)

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


the_non_monotonic = NonMonotonic(
    model_name='non_monotonic',
    baseline_utilities=baseline_utilities,
    mu_utilities=mu_utilities,
    gamma_parameters=gamma_parameters,
    alpha_parameters=alpha_parameters,
    scale_parameter=scale_parameter,
    weights=weight,
)
