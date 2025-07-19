"""

Specification of the mixtures of logit
======================================

Creation of the Biogeme object for a mixtures of logit models where
the integral is approximated using MonteCarlo integration.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 21:02:24
"""

from biogeme.biogeme import BIOGEME
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    read_data,
)
from biogeme.expressions import Beta, MonteCarlo, bioDraws, log
from biogeme.models import logit

# %%
R = 10_000


# %%
def get_biogeme(the_draws: bioDraws, number_of_draws: int) -> BIOGEME:
    """Function returning the Biogeme object as a function of the selected draws

    :param the_draws: expression representing the draws.
    :param number_of_draws: number of draws to generate.
    :return: Biogeme object.
    """

    asc_car = Beta('asc_car', 0, None, None, 0)
    asc_train = Beta('asc_train', 0, None, None, 0)
    b_time = Beta('b_time', 0, None, None, 0)
    b_time_s = Beta('b_time_s', 1, None, None, 0)
    b_cost = Beta('b_cost', 0, None, None, 0)

    # Define a random parameter, normally distributed, designed to be used
    # for Monte-Carlo simulation
    b_time_rnd = b_time + b_time_s * the_draws

    # Definition of the utility functions
    v_train = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
    v_swissmetro = b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
    v_car = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    utilities = {1: v_train, 2: v_swissmetro, 3: v_car}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # The choice model is a logit, with availability conditions
    conditional_probability = logit(utilities, av, CHOICE)
    log_probability = log(MonteCarlo(conditional_probability))

    database = read_data()

    the_biogeme = BIOGEME(database, log_probability, number_of_draws=number_of_draws)

    return the_biogeme
