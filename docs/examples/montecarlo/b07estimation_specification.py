"""

Specification of the mixtures of logit
======================================

Creation of the Biogeme object for a mixtures of logit models where
the integral is approximated using MonteCarlo integration.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 21:04:47 2023
"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, MonteCarlo, log, bioDraws
from biogeme.tools import TemporaryFile

from swissmetro import (
    database,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    SM_AV,
    CAR_AV_SP,
    CHOICE,
)

# %%
R = 2000


# %%
def get_biogeme(the_draws: bioDraws, number_of_draws: int) -> bio.BIOGEME:
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
    v_1 = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
    v_2 = b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
    v_3 = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    utilities = {1: v_1, 2: v_2, 3: v_3}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # The choice model is a logit, with availability conditions
    prob = models.logit(utilities, av, CHOICE)
    logprob = log(MonteCarlo(prob))

    with TemporaryFile() as filename:
        with open(filename, 'w', encoding='utf-8') as f:
            print('[MonteCarlo]', file=f)
            print(f'number_of_draws = {number_of_draws}', file=f)

        the_biogeme = bio.BIOGEME(database, logprob, parameter_file=filename)

    return the_biogeme
