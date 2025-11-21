from biogeme.expressions import (
    Beta,
    OrderedProbit,
)

from optima import number_of_cars
from structural_equations import car_centric_attitude, urban_preference_attitude

# %%
# Measurement equations.

# %%
# Intercept.
meas_number_cars_intercept = Beta('meas_number_cars_intercept', 0, None, None, 0)

# %%
# Coefficient for the car centric attitude latent variable
meas_number_cars_car_coefficient = Beta(
    'meas_number_cars_car_coefficient', 0, None, None, 0
)


# %%
# coefficient for the urban preference attitude latent variable
meas_number_cars_urban_coefficient = Beta(
    'meas_number_cars_urban_coefficient', 0, None, None, 0
)

# %%
# Scale parameters of the error term.
meas_number_cars_sigma_star = Beta('meas_number_cars_sigma_star', 1, 1.0e-4, None, 0)

# %%
# Contribution of the latent variables to the measurement equations.

model = (
    meas_number_cars_intercept
    + meas_number_cars_car_coefficient * car_centric_attitude
    + meas_number_cars_urban_coefficient * urban_preference_attitude
)

# %%
# Create the mapping on the Likert scale

# Symmetric threshold.
tau_1 = Beta('meas_number_cars_tau_1', 0.3, 1.0e-3, None, 0)
delta_1 = Beta('meas_number_cars_delta_1', 0.3, 1.0e-3, None, 0)
delta_2 = Beta('meas_number_cars_delta_2', 0.8, 1.0e-3, None, 0)
tau_2 = tau_1 + delta_1
tau_3 = tau_2 + delta_2

number_cars_likelihood = OrderedProbit(
    eta=model / meas_number_cars_sigma_star,
    cutpoints=[
        tau_1 / meas_number_cars_sigma_star,
        tau_2 / meas_number_cars_sigma_star,
        tau_3 / meas_number_cars_sigma_star,
    ],
    y=number_of_cars,
    categories=[0, 1, 2, 3],
    neutral_labels=[-1],
    enforce_order=True,
    eps=1e-12,
)
