"""File: 05normalMixtureMonteCarlo.py

 Author: Michel Bierlaire, EPFL
 Date: Wed Dec 11 17:11:45 2019

Calculation of a mixtures of logit models where the integral is
calculated using numerical integration and Monte-Carlo integration
with various types of draws.

"""

# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.distributions as dist
from biogeme.expressions import Integrate, RandomVariable, MonteCarlo, bioDraws

p = pd.read_csv('swissmetro.dat', sep='\t')
# Use only the first observation (index 0)
p = p.drop(p[p.index != 0].index)
database = db.Database('swissmetro', p)

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Parameters
ASC_CAR = 0.137
ASC_TRAIN = -0.402
ASC_SM = 0
B_TIME = -2.26
B_TIME_S = 1.66
B_COST = -1.29

omega = RandomVariable('omega')
density = dist.normalpdf(omega)
B_TIME_RND = B_TIME + B_TIME_S * omega
B_TIME_RND_normal = B_TIME + B_TIME_S * bioDraws('B_NORMAL', 'NORMAL')
B_TIME_RND_anti = B_TIME + B_TIME_S * bioDraws('B_ANTI', 'NORMAL_ANTI')
B_TIME_RND_halton = B_TIME + B_TIME_S * bioDraws('B_HALTON', 'NORMAL_HALTON2')
B_TIME_RND_mlhs = B_TIME + B_TIME_S * bioDraws('B_MLHS', 'NORMAL_MLHS')
B_TIME_RND_antimlhs = B_TIME + B_TIME_S * bioDraws(
    'B_ANTIMLHS', 'NORMAL_MLHS_ANTI'
)

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)
CAR_AV_SP = CAR_AV * (SP != 0)
TRAIN_AV_SP = TRAIN_AV * (SP != 0)
TRAIN_TT_SCALED = TRAIN_TT / 100.0
TRAIN_COST_SCALED = TRAIN_COST / 100
SM_TT_SCALED = SM_TT / 100.0
SM_COST_SCALED = SM_COST / 100
CAR_TT_SCALED = CAR_TT / 100
CAR_CO_SCALED = CAR_CO / 100


def logit(THE_B_TIME_RND):
    """
    Calculate the conditional logit model for a given random parameter.
    """
    V1 = (
        ASC_TRAIN
        + THE_B_TIME_RND * TRAIN_TT_SCALED
        + B_COST * TRAIN_COST_SCALED
    )
    V2 = ASC_SM + THE_B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
    V3 = ASC_CAR + THE_B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

    # Associate utility functions with the numbering of alternatives
    V = {1: V1, 2: V2, 3: V3}

    # Associate the availability conditions with the alternatives
    av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

    # The choice model is a logit, with availability conditions
    integrand = models.logit(V, av, CHOICE)
    return integrand


numericalI = Integrate(logit(B_TIME_RND) * density, 'omega')
normal = MonteCarlo(logit(B_TIME_RND_normal))
anti = MonteCarlo(logit(B_TIME_RND_anti))
halton = MonteCarlo(logit(B_TIME_RND_halton))
mlhs = MonteCarlo(logit(B_TIME_RND_mlhs))
antimlhs = MonteCarlo(logit(B_TIME_RND_antimlhs))

simulate = {
    'Numerical': numericalI,
    'MonteCarlo': normal,
    'Antithetic': anti,
    'Halton': halton,
    'MLHS': mlhs,
    'Antithetic MLHS': antimlhs,
}

R = 20000
biogeme = bio.BIOGEME(database, simulate, numberOfDraws=R)
results = biogeme.simulate()
print(f'Number of draws: {R}')
for c in results.columns:
    print(f'{c}:\t{results.loc[0,c]}')
