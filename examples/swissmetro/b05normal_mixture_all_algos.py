"""File b05normal_mixture_all_algos.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:38:34 2023

 Example of a mixture of logit models, using Monte-Carlo integration.
Estimation using several algorithms
"""

import biogeme.logging as blog
import biogeme.biogeme as bio
from biogeme import models
import biogeme.exceptions as excep
from biogeme.expressions import Beta, bioDraws, log, MonteCarlo
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
logger.info('Example b05normal_mixture_all_algos.py')

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
B_TIME = Beta('B_TIME', 0, None, None, 0)

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Conditional to B_TIME_RND, we have a logit model (called the kernel)
prob = models.logit(V, av, CHOICE)

# We integrate over B_TIME_RND using Monte-Carlo
logprob = log(MonteCarlo(prob))

algos = {
    'scipy                ': 'scipy',
    'Line search          ': 'LS-newton',
    'Trust region (dogleg)': 'TR-newton',
    'Trust region (cg)    ': 'TR-newton',
    'LS-BFGS              ': 'LS-BFGS',
    'TR-BFGS              ': 'TR-BFGS',
    'Simple bounds Newton fCG': 'simple_bounds',
    'Simple bounds BFGS fCG  ': 'simple_bounds',
    'Simple bounds hybrid fCG': 'simple_bounds',
    'Simple bounds Newton iCG': 'simple_bounds',
    'Simple bounds BFGS iCG  ': 'simple_bounds',
    'Simple bounds hybrid iCG': 'simple_bounds',
}

algoParameters = {
    'Trust region (dogleg)': {'dogleg': True},
    'Trust region (cg)    ': {'dogleg': False},
    'Simple bounds Newton fCG': {
        'second_derivatives': 1.0,
        'infeasible_cg': False,
    },
    'Simple bounds BFGS fCG  ': {
        'second_derivatives': 0.0,
        'infeasible_cg': False,
    },
    'Simple bounds hybrid fCG': {
        'second_derivatives': 0.5,
        'infeasible_cg': False,
    },
    'Simple bounds Newton iCG': {
        'second_derivatives': 1.0,
        'infeasible_cg': True,
    },
    'Simple bounds BFGS iCG  ': {
        'second_derivatives': 0.0,
        'infeasible_cg': True,
    },
    'Simple bounds hybrid iCG': {
        'second_derivatives': 0.5,
        'infeasible_cg': True,
    },
}

results = {}
msg = str()
for name, algo in algos.items():
    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database, logprob)
    the_biogeme.algorithm_name = algo
    the_biogeme.modelName = f'b05normal_mixture_all_algos_{name}'.strip()
    p = algoParameters.get(name)
    if p is not None:
        for attr, value in p.items():
            setattr(the_biogeme, attr, value)
    try:
        results[name] = the_biogeme.estimate()
        msg += (
            f'{name}\t{results[name].data.logLike:.2f}\t'
            f'{results[name].data.gradientNorm:.2g}\t'
            f'{results[name].data.optimizationMessages["Optimization time"]}'
            f'\t{results[name].data.optimizationMessages["Cause of termination"]}'
            f'\n'
        )
    except excep.BiogemeError as e:
        print(e)
        results[name] = None
        msg += f'{name}\tFailed to estimate the model'


SUMMARY_FILE = '05normalMixture_allAlgos.log'
with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    print('Algorithm\t\tloglike\t\tnormg\ttime\t\tdiagnostic', file=f)
    print('+++++++++\t\t+++++++\t\t+++++\t++++\t\t++++++++++', file=f)
    print(msg, file=f)
print(f'Summary reported in file {SUMMARY_FILE}')
