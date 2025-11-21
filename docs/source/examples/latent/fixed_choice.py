"""

Estimation of the choice model
==============================

Choice model without any latent variable.

Michel Bierlaire, EPFL
Wed Sept 03 2025, 08:18:01

"""

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, exp
from biogeme.models import boxcox, loglogit
from biogeme.results_processing import (
    EstimationResults,
)

from fixed_latent import car_centric_attitude, urban_preference_attitude
from optima import (
    Choice,
    CostCarCHF,
    MarginalCostPT,
    PurpHWH,
    TimeCar_hour,
    TimePT_hour,
    WaitingTimePT,
    distance_km,
    distance_km_scaled,
    read_data,
)

logger = blog.get_screen_logger(level=blog.INFO)

database = read_data()


work_trip = PurpHWH == 1
other_trip_purposes = PurpHWH != 1

lambda_distance = Beta('lambda_distance', 1, -10, 10, 0)
boxcox_distance = boxcox(distance_km_scaled, lambda_distance)

# Choice model: parameters
choice_beta_cost = Beta('choice_beta_cost', -1, None, None, 1)

choice_asc_car = Beta('choice_asc_car', 0.0, None, None, 0)

choice_asc_pt = Beta('choice_asc_pt', 0, None, None, 0)

choice_beta_dist_work = Beta('choice_beta_dist_work', 0, None, None, 0)
choice_beta_dist_other_purposes = Beta(
    'choice_beta_dist_other_purposes', 0, None, None, 0
)
choice_beta_dist = (
    choice_beta_dist_work * work_trip
    + choice_beta_dist_other_purposes * other_trip_purposes
)

choice_beta_time_car = Beta('choice_beta_time_car', 0, None, None, 0)

choice_beta_time_pt = Beta('choice_beta_time_pt', 0, None, None, 0)

choice_beta_waiting_time_work = Beta('choice_beta_waiting_time_work', 0, None, None, 0)
choice_beta_waiting_time_other_purposes = Beta(
    'choice_beta_waiting_time_other_purposes', 0, None, None, 0
)
choice_beta_waiting_time = (
    choice_beta_waiting_time_work * work_trip
    + choice_beta_waiting_time_other_purposes * other_trip_purposes
)

log_scale_choice_model = Beta('log_scale_choice_model', 0, None, None, 0)

scale_choice_model = exp(log_scale_choice_model)

choice_car_centric_car_cte = Beta('choice_car_centric_car_cte', 1, None, None, 0)
choice_car_centric_pt_cte = Beta('choice_car_centric_pt_cte', 1, None, None, 0)
choice_urban_life_car_cte = Beta('choice_urban_life_car_cte', 0, None, None, 1)
choice_urban_life_pt_cte = Beta('choice_urban_life_pt_cte', 0, None, None, 1)

lambda_urban_preference = Beta('lambda_urban_preference', -1, None, None, 0)


# %%
# Definition of utility functions:
v_public_transport = scale_choice_model * (
    choice_asc_pt
    + choice_beta_time_pt * TimePT_hour
    + choice_beta_waiting_time
    * exp(lambda_urban_preference * urban_preference_attitude)
    * WaitingTimePT
    / 60
    + choice_beta_cost * MarginalCostPT
    + choice_car_centric_pt_cte * car_centric_attitude
    + choice_urban_life_pt_cte * urban_preference_attitude
)

v_car = scale_choice_model * (
    choice_asc_car
    + choice_beta_time_car * TimeCar_hour
    + choice_beta_cost * CostCarCHF
    + choice_car_centric_car_cte * car_centric_attitude
    + choice_urban_life_car_cte * urban_preference_attitude
)


v_slow_modes = scale_choice_model * (choice_beta_dist * distance_km)

# %%
# Associate utility functions with the numbering of alternatives
v = {0: v_public_transport, 1: v_car, 2: v_slow_modes}


# %%
# We integrate over omega using numerical integration
log_likelihood = loglogit(v, None, Choice)


# %%
# Create the Biogeme object
print('Create the biogeme object')
the_biogeme = BIOGEME(database, log_likelihood)
the_biogeme.model_name = 'fixed_choice'


print('--- Estimate ---')
results: EstimationResults = the_biogeme.estimate()
