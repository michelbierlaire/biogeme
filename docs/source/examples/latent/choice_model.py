"""

Specification of the choice model
=================================

Implementation of the utility functions

Michel Bierlaire, EPFL
Wed Sept 03 2025, 08:13:34

"""

from optima import (
    CostCarCHF,
    MarginalCostPT,
    TimeCar_hour,
    TimePT_hour,
    WaitingTimePT,
    distance_category,
    distance_km,
)

from biogeme.expressions import Beta, exp

# %%
# Choice model: parameters
choice_asc_car_short_dist = Beta('choice_asc_car_short_dist', 0.0, None, None, 0)
choice_asc_car_long_dist = Beta('choice_asc_car_long_dist', 0.0, None, None, 0)
choice_asc_pt_short_dist = Beta('choice_asc_pt_short_dist', 0, None, None, 0)
choice_asc_pt_long_dist = Beta('choice_asc_pt_long_dist', 0, None, None, 0)
choice_beta_cost = Beta('choice_beta_cost', -1, None, None, 1)
choice_beta_dist = Beta('choice_beta_dist', 0, None, None, 0)
choice_beta_time_car_short_dist = Beta(
    'choice_beta_time_car_short_dist', 0, None, None, 0
)
choice_beta_time_car_long_dist = Beta(
    'choice_beta_time_car_long_dist', 0, None, None, 0
)
choice_beta_time_pt_short_dist = Beta(
    'choice_beta_time_pt_short_dist', 0, None, None, 0
)
choice_beta_time_pt_long_dist = Beta('choice_beta_time_pt_long_dist', 0, None, None, 0)
choice_beta_waiting_time = Beta('choice_beta_waiting_time', 0, None, None, 0)

log_scale_choice_model_long_dist = Beta(
    'log_scale_choice_model_long_dist', 0, None, None, 0
)
log_scale_choice_model_medium_dist = Beta(
    'log_scale_choice_model_medium_dist', 0, None, None, 0
)
log_scale_choice_model_short_dist = Beta(
    'log_scale_choice_model_short_dist', 0, None, None, 0
)
log_scale_choice_model = log_scale_choice_model_short_dist * (
    distance_category == 1
) + log_scale_choice_model_long_dist * (distance_category == 2)

scale_choice_model = exp(log_scale_choice_model)

choice_asc_car = choice_asc_car_short_dist * (
    distance_category == 1
) + choice_asc_car_long_dist * (distance_category == 2)

choice_asc_pt = choice_asc_pt_short_dist * (
    distance_category == 1
) + choice_asc_pt_long_dist * (distance_category == 2)

choice_beta_time_pt = choice_beta_time_pt_short_dist * (
    distance_category == 1
) + choice_beta_time_pt_long_dist * (distance_category == 2)
choice_beta_time_car = choice_beta_time_car_short_dist * (
    distance_category == 1
) + choice_beta_time_car_long_dist * (distance_category == 2)
# %%
# Definition of utility functions:
v_public_transport = scale_choice_model * (
    choice_asc_pt
    + choice_beta_time_pt * TimePT_hour
    + choice_beta_waiting_time * WaitingTimePT / 60
    + choice_beta_cost * MarginalCostPT
)

v_car = scale_choice_model * (
    choice_asc_car + choice_beta_time_car * TimeCar_hour + choice_beta_cost * CostCarCHF
)

v_slow_modes = scale_choice_model * (choice_beta_dist * distance_km)

# %%
# Associate utility functions with the numbering of alternatives
v = {0: v_public_transport, 1: v_car, 2: v_slow_modes}
