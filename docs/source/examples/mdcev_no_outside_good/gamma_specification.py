"""File gamma_specification.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 11:09:58 2024

Specification of the "gamma_profile" MDCEV model.
"""

from biogeme.expressions import Beta
from biogeme.mdcev import GammaProfile
from specification import (
    weight,
    baseline_utilities,
)

# %
# Gamma parameters. Must be positive.
lowest_positive_value = 0.0001
gamma_shopping = Beta('gamma_shopping', 1, lowest_positive_value, None, 0)
gamma_socializing = Beta('gamma_socializing', 1, lowest_positive_value, None, 0)
gamma_recreation = Beta('gamma_recreation', 1, lowest_positive_value, None, 0)
gamma_personal = Beta('gamma_personal', 1, lowest_positive_value, None, 0)

scale_parameter = Beta('scale', 1, lowest_positive_value, None, 0)

gamma_parameters = {
    1: gamma_shopping,
    2: gamma_socializing,
    3: gamma_recreation,
    4: gamma_personal,
}

the_gamma_profile = GammaProfile(
    model_name='gamma_profile',
    baseline_utilities=baseline_utilities,
    gamma_parameters=gamma_parameters,
    scale_parameter=scale_parameter,
    weights=weight,
)
