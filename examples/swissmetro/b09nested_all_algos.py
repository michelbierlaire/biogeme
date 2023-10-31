"""File b09nested_allAlgos.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 18:01:30 2023

 Example of a nested logit model. Estimation with several algorithms
"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta

from swissmetro_data import (
    database,
    CHOICE,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b09nested_allAlgos.py')


# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
MU = Beta('MU', 1, 1, 10, 0)

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests:
# 1: nests parameter
# 2: list of alternatives
existing = MU, [1, 3]
future = 1.0, [2]
nests = existing, future

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
# The choice model is a nested logit, with availability conditions
logprob = models.lognested(V, av, nests, CHOICE)

algos = {
    'scipy                   ': 'scipy',
    'Simple bounds Newton    ': 'simple_bounds',
    'Simple bounds BFGS      ': 'simple_bounds',
    'Simple bounds hybrid 20%': 'simple_bounds',
    'Simple bounds hybrid 50%': 'simple_bounds',
    'Simple bounds hybrid 80%': 'simple_bounds',
}

algoParameters = {
    'Simple bounds Newton    ': {'second_derivatives': 1.0},
    'Simple bounds BFGS      ': {'second_derivatives': 0.0},
    'Simple bounds hybrid 20%': {'second_derivatives': 0.2},
    'Simple bounds hybrid 50%': {'second_derivatives': 0.5},
    'Simple bounds hybrid 80%': {'second_derivatives': 0.8},
}

results = {}
msg = ''
for name, algo in algos.items():
    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database, logprob)
    the_biogeme.modelName = f'b09nested_all_algos_{name}'.strip()
    the_biogeme.algorithm_name = algo
    p = algoParameters.get(name)
    if p is not None:
        for attr, value in p.items():
            setattr(the_biogeme, attr, value)
    results[name] = the_biogeme.estimate()
    msg += (
        f'{name}\t{results[name].data.logLike:.2f}\t'
        f'{results[name].data.gradientNorm:.2g}\t'
        f'{results[name].data.optimizationMessages["Optimization time"]}'
        f'\t{results[name].data.optimizationMessages["Cause of termination"]}'
        f'\n'
    )

print("Algorithm\t\t\tloglike\t\tnormg\ttime\t\tdiagnostic")
print("+++++++++\t\t\t+++++++\t\t+++++\t++++\t\t++++++++++")
print(msg)
