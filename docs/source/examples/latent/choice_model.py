"""

Specification of the choice model
=================================

Implementation of the utility functions

Michel Bierlaire, EPFL
Wed Sept 03 2025, 08:13:34

"""

from biogeme.expressions import Beta, exp
from biogeme.models import boxcox
from biogeme.segmentation import Segmentation

from optima import (
    CostCarCHF,
    LangCode,
    MarginalCostPT,
    OccupStat,
    PurpHWH,
    TimeCar_hour,
    TimePT_hour,
    WaitingTimePT,
    car_is_available,
    distance_km,
    distance_km_scaled,
    haveGA,
    male,
    read_data,
)

model_name = 'choice_01'

database = read_data()


trip_purpose_segmentation = database.generate_segmentation(
    variable=PurpHWH,
    mapping={1: 'work_trip', 0: 'other_trip_purposes'},
    reference='work_trip',
)

language_segmentation = database.generate_segmentation(
    variable=LangCode, mapping={1: 'french', 2: 'german'}, reference='german'
)

car_availability_segmentation = database.generate_segmentation(
    variable=car_is_available,
    mapping={1: 'car_avail', 0: 'car_unavail'},
    reference='car_avail',
)

gender_segmentation = database.generate_segmentation(
    variable=male, mapping={0: 'female', 1: 'male'}, reference='male'
)

ga_segmentation = database.generate_segmentation(
    variable=haveGA, mapping={0: 'without_ga', 1: 'with_ga'}, reference='without_ga'
)

occup_segmentation = database.generate_segmentation(
    variable=OccupStat,
    mapping={1: 'full_time', 2: 'part_time'},
    reference='full_time',
)

# piecewise_distance = piecewise_as_variable(
#    distance_km, [None, 5, 10, 30, 60, None], betas=None
# )
lambda_distance = Beta('lambda_distance', 1, -10, 10, 0)
boxcox_distance = boxcox(distance_km_scaled, lambda_distance)
# %%
choice_asc_car_segmentations = []
choice_asc_pt_segmentations = []
choice_beta_dist_segmentations = [trip_purpose_segmentation]
choice_beta_time_car_segmentations = []
choice_beta_time_pt_segmentations = []
choice_beta_waiting_time_segmentations = [trip_purpose_segmentation]
log_scale_choice_model_segmentations = []


# Choice model: parameters
choice_beta_cost = Beta('choice_beta_cost', -1, None, None, 1)

choice_asc_car = Beta('choice_asc_car', 0.0, None, None, 0)
segmented_choice_asc_car = (
    Segmentation(choice_asc_car, choice_asc_car_segmentations).segmented_beta()
    if choice_asc_car_segmentations
    else choice_asc_car
)

choice_asc_pt = Beta('choice_asc_pt', 0, None, None, 0)
segmented_choice_asc_pt = (
    Segmentation(choice_asc_pt, choice_asc_pt_segmentations).segmented_beta()
    if choice_asc_pt_segmentations
    else choice_asc_pt
)


choice_beta_dist = Beta('choice_beta_dist', 0, None, None, 0)
segmented_choice_beta_dist = (
    Segmentation(
        choice_beta_dist,
        choice_beta_dist_segmentations,
    ).segmented_beta()
    if choice_beta_dist_segmentations
    else choice_beta_dist
)

choice_beta_time_car = Beta('choice_beta_time_car', 0, None, None, 0)
segmented_choice_beta_time_car = (
    Segmentation(
        choice_beta_time_car, choice_beta_time_car_segmentations
    ).segmented_beta()
    if choice_beta_time_car_segmentations
    else choice_beta_time_car
)

choice_beta_time_pt = Beta('choice_beta_time_pt', 0, None, None, 0)
segmented_choice_beta_time_pt = (
    Segmentation(
        choice_beta_time_pt, choice_beta_time_pt_segmentations
    ).segmented_beta()
    if choice_beta_time_pt_segmentations
    else choice_beta_time_pt
)

choice_beta_waiting_time = Beta('choice_beta_waiting_time', 0, None, None, 0)
segmented_choice_beta_waiting_time = (
    Segmentation(
        choice_beta_waiting_time, choice_beta_waiting_time_segmentations
    ).segmented_beta()
    if choice_beta_waiting_time_segmentations
    else choice_beta_waiting_time
)

log_scale_choice_model = Beta('log_scale_choice_model', 0, None, None, 0)

segmented_log_scale_choice_model = (
    Segmentation(
        log_scale_choice_model,
        log_scale_choice_model_segmentations,
    ).segmented_beta()
    if log_scale_choice_model_segmentations
    else log_scale_choice_model
)

segmented_scale_choice_model = exp(segmented_log_scale_choice_model)

# %%
# Definition of utility functions:
v_public_transport = segmented_scale_choice_model * (
    segmented_choice_asc_pt
    + segmented_choice_beta_time_pt * TimePT_hour
    + segmented_choice_beta_waiting_time * WaitingTimePT / 60
    + choice_beta_cost * MarginalCostPT
)

v_car = segmented_scale_choice_model * (
    segmented_choice_asc_car
    + segmented_choice_beta_time_car * TimeCar_hour
    + choice_beta_cost * CostCarCHF
)

v_slow_modes = segmented_scale_choice_model * (segmented_choice_beta_dist * distance_km)

# %%
# Associate utility functions with the numbering of alternatives
v = {0: v_public_transport, 1: v_car, 2: v_slow_modes}

value_of_time_pt_chf_hour = -segmented_choice_beta_time_pt
value_of_time_car_chf_hour = -segmented_choice_beta_time_car
