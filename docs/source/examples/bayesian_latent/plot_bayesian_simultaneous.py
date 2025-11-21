"""

Choice model with latent variables: simultaneous estimation
===========================================================

Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

Michel Bierlaire, EPFL
Fri May 16 2025, 15:53:52
"""

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.bayesian_estimation import get_pandas_estimated_parameters
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Beta,
    exp,
)
from biogeme.models import boxcox, loglogit

from measurement_equations import likert_log_likelihood_indicator
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
from read_or_estimate import read_or_estimate
from structural_equations import car_centric_attitude, urban_preference_attitude

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Choice model


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

# %%
# Alternative specific constants
choice_car_centric_car_cte = Beta('choice_car_centric_car_cte', 1, None, None, 0)
choice_car_centric_pt_cte = Beta('choice_car_centric_pt_cte', 1, None, None, 0)
choice_urban_life_car_cte = Beta('choice_urban_life_car_cte', 1, None, None, 0)
choice_urban_life_pt_cte = Beta('choice_urban_life_pt_cte', 1, None, None, 0)

# %%
# Definition of utility functions:
v_public_transport = scale_choice_model * (
    choice_asc_pt
    + choice_beta_time_pt * TimePT_hour
    + choice_beta_waiting_time * WaitingTimePT / 60
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
# Conditional on the latent variables, we have a logit model (called the kernel)
cond_log_prob = loglogit(v, None, Choice) + likert_log_likelihood_indicator

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(
    database,
    cond_log_prob,
    warmup=10_000,
    bayesian_draws=10_000,
    calculate_likelihood=False,
    calculate_waic=False,
    calculate_loo=False,
)
the_biogeme.model_name = 'bayesian_simultaneous'

results = read_or_estimate(the_biogeme=the_biogeme)

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
