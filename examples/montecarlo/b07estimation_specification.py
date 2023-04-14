""" File: b07estimation_specification.py

 Author: Michel Bierlaire, EPFL
 Date: Thu Apr 13 21:04:47 2023

Estimation of a mixtures of logit models where the integral is
approximated using MonteCarlo integration.

"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, MonteCarlo, log

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

R = 2000


def get_biogeme(the_draws):
    """Function returning the Biogeme object as a function of the selected draws

    :param the_draws: expression representing the draws
    :type the_draws: biogeme.expressions.bioDraws
    """

    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)

    # Define a random parameter, normally distributed, designed to be used
    # for Monte-Carlo simulation
    b_time_rnd = B_TIME + B_TIME_S * the_draws

    # Definition of the utility functions
    v_1 = ASC_TRAIN + b_time_rnd * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
    v_2 = b_time_rnd * SM_TT_SCALED + B_COST * SM_COST_SCALED
    v_3 = ASC_CAR + b_time_rnd * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    V = {1: v_1, 2: v_2, 3: v_3}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # The choice model is a logit, with availability conditions
    prob = models.logit(V, av, CHOICE)
    logprob = log(MonteCarlo(prob))

    the_biogeme = bio.BIOGEME(database, logprob)
    return the_biogeme
