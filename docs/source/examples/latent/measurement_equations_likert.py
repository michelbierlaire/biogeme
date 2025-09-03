"""

Specification of the discrete measurement equations
===================================================

Measurement equations for the Likert indicators as functions of the latent variables.

Michel Bierlaire
Wed Sept 03 2025, 08:15:52
"""

import numpy as np
from likelihood_discrete import likelihood_discrete_mimic
from measurement_equations_continuous import generate_continuous_measurement_equations
from relevant_data import (
    latent_variables_indicators,
    normalized,
)
from structural_equations import LatentVariable

from biogeme.expressions import Beta, Expression, exp

# %%
# Measurement equations.


def generate_likert_measurement_equations(
    car_centric_attitude: LatentVariable, urban_preference_attitude: LatentVariable
) -> Expression:
    """

    :param car_centric_attitude: expression for the latent_old variable.
    :param urban_preference_attitude: expression for the latent_old variable.
    :return: the likelihood contribution of the likert indicators
    """
    models = generate_continuous_measurement_equations(
        latent_variables=[car_centric_attitude, urban_preference_attitude],
        latent_variables_indicators=latent_variables_indicators,
        normalized=normalized,
    )

    # Symmetric threshold.
    delta_1 = exp(Beta('log_delta_1', np.log(0.3), None, None, 0))
    delta_2 = exp(Beta('log_delta_2', np.log(0.8), None, None, 0))
    tau_1 = -delta_1 - delta_2
    tau_2 = -delta_1
    tau_3 = delta_1
    tau_4 = delta_1 + delta_2

    likert_likelihood = likelihood_discrete_mimic(
        measurement_equations=models,
        threshold_parameters=[tau_1, tau_2, tau_3, tau_4],
        discrete_values=[1, 2, 3, 4, 5],
        missing_values=[6, -1, -2],
    )
    return likert_likelihood
