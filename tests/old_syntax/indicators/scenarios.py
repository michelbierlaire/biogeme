"""
.. _scenarios:

Specification of a nested logit model
=====================================

Specification of a nested logit model, that will be estimated, and
 used for simulation.  Three alternatives: public transporation, car
 and slow modes.  RP data.  Based on the Optima data.  It contains a
 function that generates scenarios where the current cost of public
 transportation is multiplied by a factor.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 20:51:37 2023

"""

from biogeme.expressions import Expression, Beta
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
from optima_data import (
    Choice,
    TimePT,
    TimeCar,
    CostCarCHF,
    MarginalCostPT,
    distance_km,
    Gender,
    OccupStat,
)

# %%
# List of parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0, None, None, 0)
BETA_TIME_FULLTIME = Beta('BETA_TIME_FULLTIME', 0, None, None, 0)
BETA_TIME_OTHER = Beta('BETA_TIME_OTHER', 0, None, None, 0)
BETA_DIST_MALE = Beta('BETA_DIST_MALE', 0, None, None, 0)
BETA_DIST_FEMALE = Beta('BETA_DIST_FEMALE', 0, None, None, 0)
BETA_DIST_UNREPORTED = Beta('BETA_DIST_UNREPORTED', 0, None, None, 0)
BETA_COST = Beta('BETA_COST', 0, None, None, 0)

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
    :type factor: float

    :return: a dict with the utility functions, the nesting structure,
        and the choice expression.

    :rtype: dict(int: biogeme.expression), tuple(biogeme.expression,
        list(int)), biogeme.expression
    """
    marginal_cost_scenario = MarginalCostPT * factor
    marginal_cost_pt_scaled = marginal_cost_scenario / 10
    # Definition of utility functions:
    v_pt = (
        ASC_PT
        + BETA_TIME_FULLTIME * TimePT_scaled * fulltime
        + BETA_TIME_OTHER * TimePT_scaled * notfulltime
        + BETA_COST * marginal_cost_pt_scaled
    )
    v_car = (
        ASC_CAR
        + BETA_TIME_FULLTIME * TimeCar_scaled * fulltime
        + BETA_TIME_OTHER * TimeCar_scaled * notfulltime
        + BETA_COST * CostCarCHF_scaled
    )
    v_sm = (
        ASC_SM
        + BETA_DIST_MALE * distance_km_scaled * male
        + BETA_DIST_FEMALE * distance_km_scaled * female
        + BETA_DIST_UNREPORTED * distance_km_scaled * unreportedGender
    )

    # Associate utility functions with the numbering of alternatives
    V = {0: v_pt, 1: v_car, 2: v_sm}

    # Definition of the nests:
    # 1: nests parameter
    # 2: list of alternatives
    mu_nocar = Beta('mu_nocar', 1.0, 1.0, None, 0)

    no_car_nest = OneNestForNestedLogit(
        nest_param=mu_nocar, list_of_alternatives=[0, 2], name='no_car'
    )
    nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(no_car_nest,))
    return V, nests, Choice, marginal_cost_scenario
