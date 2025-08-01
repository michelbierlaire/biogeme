"""
.. _scenarios:

Specification of a nested logit model
=====================================

Specification of a nested logit model, that will be estimated, and
 used for simulation.  Three alternatives: public transportation, car
 and slow modes.  RP data.  Based on the Optima data.  It contains a
 function that generates scenarios where the current cost of public
 transportation is multiplied by a factor.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 15:52:11
"""

from biogeme.data.optima import (
    Choice,
    CostCarCHF,
    Gender,
    MarginalCostPT,
    OccupStat,
    TimeCar,
    TimePT,
    distance_km,
)
from biogeme.expressions import Beta, Expression
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit

# %%
# List of parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_pt = Beta('asc_pt', 0, None, None, 1)
asc_sm = Beta('asc_sm', 0, None, None, 0)
beta_time_fulltime = Beta('beta_time_fulltime', 0, None, None, 0)
beta_time_other = Beta('beta_time_other', 0, None, None, 0)
beta_dist_male = Beta('beta_dist_male', 0, None, None, 0)
beta_dist_female = Beta('beta_dist_female', 0, None, None, 0)
beta_dist_unreported = Beta('beta_dist_unreported', 0, None, None, 0)
beta_cost = Beta('beta_cost', 0, None, None, 0)

# %%
# Definition of variables:
# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0.
TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5
male = Gender == 1
female = Gender == 2
unreportedGender = Gender == -1

fulltime = OccupStat == 1
notfulltime = OccupStat != 1


# %%
# Model specification as a function of the multiplication factor for
# the price of public transportation.
def scenario(
    factor: float = 1.0,
) -> tuple[dict[int, Expression], NestsForNestedLogit, Expression, float]:
    """Provide the model specification for a scenario with the price of
        public transportation is multiplied by a factor

    :param factor: factor that multiples the price of public transportation.
    :return: a dict with the utility functions, the nesting structure,
        and the choice expression.

    """
    marginal_cost_scenario = MarginalCostPT * factor
    marginal_cost_pt_scaled = marginal_cost_scenario / 10
    # Definition of utility functions:
    v_pt = (
        asc_pt
        + beta_time_fulltime * TimePT_scaled * fulltime
        + beta_time_other * TimePT_scaled * notfulltime
        + beta_cost * marginal_cost_pt_scaled
    )
    v_car = (
        asc_car
        + beta_time_fulltime * TimeCar_scaled * fulltime
        + beta_time_other * TimeCar_scaled * notfulltime
        + beta_cost * CostCarCHF_scaled
    )
    v_sm = (
        asc_sm
        + beta_dist_male * distance_km_scaled * male
        + beta_dist_female * distance_km_scaled * female
        + beta_dist_unreported * distance_km_scaled * unreportedGender
    )

    # Associate utility functions with the numbering of alternatives
    v = {0: v_pt, 1: v_car, 2: v_sm}

    # Definition of the nests:
    # 1: nests parameter
    # 2: list of alternatives
    mu_no_car = Beta('mu_no_car', 1, 1, 2, 0)

    no_car_nest = OneNestForNestedLogit(
        nest_param=mu_no_car, list_of_alternatives=[0, 2], name='no_car'
    )
    car_nest = OneNestForNestedLogit(
        nest_param=1.0, list_of_alternatives=[1], name='car'
    )
    nests = NestsForNestedLogit(
        choice_set=list(v), tuple_of_nests=(no_car_nest, car_nest)
    )
    return v, nests, Choice, marginal_cost_scenario
