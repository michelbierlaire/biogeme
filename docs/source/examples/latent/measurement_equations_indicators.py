from biogeme.expressions import (
    Beta,
    MultipleProduct,
    OrderedProbit,
    Variable,
)

from structural_equations import car_centric_attitude, urban_preference_attitude

# %%
# Measurement equations.

# %%
# indicators for the car centric attitude
car_likert_indicators = {
    'Envir01',
    'Envir02',
    'Envir06',
    'Mobil03',
    'Mobil04',
    'Mobil05',
    'Mobil06',
    'Mobil07',
    'Mobil08',
    'Mobil09',
    'Mobil10',
    'LifSty07',
    'LifSty08',
}

# %%
# indicators for the urban preference attitude
urban_likert_indicators = {
    'ResidCh01',
    'ResidCh02',
    'ResidCh03',
    'ResidCh05',
    'ResidCh06',
    'ResidCh07',
    'LifSty07',
}

# %%
# Set of indicators
all_indicators = car_likert_indicators | urban_likert_indicators
common_indicators = car_likert_indicators & urban_likert_indicators
only_car_indicators = car_likert_indicators - common_indicators
only_urban_indicators = urban_likert_indicators - common_indicators

# %%
# Intercepts. One per indicator
intercepts: dict[str, Beta | float] = {
    k: Beta(f'meas_intercept_{k}', 0, None, None, 0) for k in all_indicators
}

# %%
# Coefficients for the car centric attitude latent variable
car_coefficients: dict[str, Beta | float] = {
    k: Beta(f'car_meas_b_{k}', 0, None, None, 0) for k in car_likert_indicators
}

# %%
# The indicator that is used to normalize the latent variable "car centric attitude" is
# "Fuel price should be increased to reduce congestion and air"
# Therefore, we normalize the coefficient to -1.
normalized_car = 'Envir01'
car_coefficients[normalized_car] = -1.0

# %%
# coefficients for the urban preference attitude latent variable
urban_coefficients: dict[str, Beta | float] = {
    k: Beta(f'urban_meas_b_{k}', 0, None, None, 0) for k in urban_likert_indicators
}

# %%
# Normalization
normalized_urban = 'ResidCh01'
urban_coefficients[normalized_urban] = 1.0

# %%
# Scale parameters of the error terms.
sigma_star: dict[str, Beta | float] = {
    k: Beta(f'meas_sigma_star_{k}', 1, 1.0e-4, None, 0) for k in all_indicators
}

# %%
# Normalization of the scale parameters
sigma_star[normalized_car] = 1.0
sigma_star[normalized_urban] = 1.0


# %%
# Contribution of the latent variables to the measurement equations.

# %%
# Indicators involving only the car centric latent variable
models = {
    indicator: intercepts[indicator]
    + car_coefficients[indicator] * car_centric_attitude
    for indicator in only_car_indicators
}

# %%
# Indicators involving only the urban preference latent variable
models |= {
    indicator: intercepts[indicator]
    + urban_coefficients[indicator] * urban_preference_attitude
    for indicator in only_urban_indicators
}

# %%
# Indicators involving both latent variables
models |= {
    indicator: intercepts[indicator]
    + car_coefficients[indicator] * car_centric_attitude
    + urban_coefficients[indicator] * urban_preference_attitude
    for indicator in common_indicators
}

# %%
# Create the mapping on the Likert scale

# Symmetric threshold.
delta_1 = Beta('delta_1', 0.3, 1.0e-3, None, 0)
delta_2 = Beta('delta_2', 0.8, 1.0e-3, None, 0)
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1
tau_3 = delta_1
tau_4 = delta_1 + delta_2

ordered_ll = {
    indicator: OrderedProbit(
        eta=models[indicator] / sigma_star[indicator],
        cutpoints=[
            tau_1 / sigma_star[indicator],
            tau_2 / sigma_star[indicator],
            tau_3 / sigma_star[indicator],
            tau_4 / sigma_star[indicator],
        ],
        y=Variable(indicator),
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[6, -1, -2],
        enforce_order=True,
        eps=1e-12,
    )
    for indicator in all_indicators
}


likert_likelihood_indicator = MultipleProduct(ordered_ll)
