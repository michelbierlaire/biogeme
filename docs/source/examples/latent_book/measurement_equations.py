from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression, NormalCdf
from relevant_data import (
    car_indicators,
    normalized_car,
    normalized_urban,
    urban_indicators,
)

# %%
# Measurement equations.

# %%
# All indicators
all_indicators = car_indicators | urban_indicators


# %%
# Intercepts
intercepts: dict[str, Beta | float] = {
    k: Beta(f'meas_intercept_{k}', 0, None, None, 0) for k in all_indicators
}

# %%
# Normalization of the intercepts
intercepts[normalized_car] = 0.0
intercepts[normalized_urban] = 0.0

# %%
# coefficients for the car centric attitude latent variable
car_coefficients: dict[str, Beta | float] = {
    k: Beta(f'car_meas_b_{k}', 0, None, None, 0) for k in car_indicators
}

# %%
# The indicator that is used to normalize the latent variable "car centric attitude" is
# "Fuel price should be increased to reduce congestion and air"
# Therefore, we normalize the coefficient to -1.
car_coefficients[normalized_car] = -1.0

# %%
# coefficients for the urban preference attitude latent variable
urban_coefficients: dict[str, Beta | float] = {
    k: Beta(f'urban_meas_b_{k}', 0, None, None, 0) for k in urban_indicators
}
urban_coefficients[normalized_urban] = 1.0

# %%
# Scale parameters of the error terms.
sigma_star: dict[str, Beta | float] = {
    k: Beta(f'meas_sigma_star_{k}', 10, 1.0e-5, None, 0) for k in all_indicators
}

# %%
# Normalization of the scale parameters
sigma_star[normalized_car] = 1
sigma_star[normalized_urban] = 1


# %%
# Contribution of the latent variables to the measurement equations
def generate_model_terms(
    indicator: str,
    car_centric_attitude: Expression,
    urban_preference_attitude: Expression,
) -> Expression:
    """Returns the contribution of the latent variables to the measurement equation
    for a given indicator."""
    is_car = indicator in car_indicators
    is_urban = indicator in urban_indicators

    if is_car and is_urban:
        # Indicator is influenced by both latent variables
        return (
            car_coefficients[indicator] * car_centric_attitude
            + urban_coefficients[indicator] * urban_preference_attitude
        )
    if is_car:
        # Indicator is influenced by the car-centric latent variable
        return car_coefficients[indicator] * car_centric_attitude
    if is_urban:
        # Indicator is influenced by the urban preference latent variable
        return urban_coefficients[indicator] * urban_preference_attitude
    raise BiogemeError(f'Unknown indicator: {indicator}')


def generate_measurement_equations(
    car_centric_attitude: Expression, urban_preference_attitude: Expression
) -> dict[int, Expression | float]:
    """

    :param structural_parameters:
    :return: a dict associating each level of the Liker scale with the Expression calculating the corresponding probability.

    The first one is normalized to -1. Indeed, we expect the respondents with a higher car
    # centric attitude to disagree more with the statement
    # "Fuel price should be increased to reduce congestion and air pollution."
    # (corresponding to a low value of the indicator) compared to those who have a lower car centric attitude. In other
    # words, when the car centric attitude increases, we expect the value of the indicator to decrease.
    """
    #
    models = {
        k: intercepts[k]
        + generate_model_terms(
            k,
            car_centric_attitude=car_centric_attitude,
            urban_preference_attitude=urban_preference_attitude,
        )
        for k in all_indicators
    }

    # Symmetric threshold.
    delta_1 = Beta('delta_1', 0.1, 1.0e-5, None, 0)
    delta_2 = Beta('delta_2', 0.2, 1.0e-5, None, 0)
    tau_1 = -delta_1 - delta_2
    tau_2 = -delta_1
    tau_3 = delta_1
    tau_4 = delta_1 + delta_2

    # %%
    # Ordered probit models.
    tau_1_residual = {
        indicator: (tau_1 - models[indicator]) / sigma_star[indicator]
        for indicator in all_indicators
    }
    tau_2_residual = {
        indicator: (tau_2 - models[indicator]) / sigma_star[indicator]
        for indicator in all_indicators
    }
    tau_3_residual = {
        indicator: (tau_3 - models[indicator]) / sigma_star[indicator]
        for indicator in all_indicators
    }
    tau_4_residual = {
        indicator: (tau_4 - models[indicator]) / sigma_star[indicator]
        for indicator in all_indicators
    }
    dict_prob_indicators = {
        indicator: {
            1: NormalCdf(tau_1_residual[indicator]),
            2: NormalCdf(tau_2_residual[indicator])
            - NormalCdf(tau_1_residual[indicator]),
            3: NormalCdf(tau_3_residual[indicator])
            - NormalCdf(tau_2_residual[indicator]),
            4: NormalCdf(tau_4_residual[indicator])
            - NormalCdf(tau_3_residual[indicator]),
            5: 1 - NormalCdf(tau_4_residual[indicator]),
            6: 1.0,
            -1: 1.0,
            -2: 1.0,
        }
        for indicator in all_indicators
    }
    return dict_prob_indicators
