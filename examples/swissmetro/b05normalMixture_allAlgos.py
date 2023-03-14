"""File 05normalMixture_allAlgos.py

:author: Michel Bierlaire, EPFL
:date: Tue Dec  6 18:14:59 2022

 Example of a mixture of logit models, using Monte-Carlo integration.
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
import biogeme.exceptions as excep
from biogeme.expressions import Beta, bioDraws, log, MonteCarlo
from swissmetro import (
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

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()

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
msg = ''
for name, algo in algos.items():
    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob, parameter_file='draws.toml')
    biogeme.algorithm_name = algo
    biogeme.modelName = f'05normalMixture_allAlgos_{name}'.strip()
    p = algoParameters.get(name)
    if p is not None:
        for attr, value in p.items():
            setattr(biogeme, attr, value)
    try:
        results[name] = biogeme.estimate()
        msg += (
            f'{name}\t{results[name].data.logLike:.2f}\t'
            f'{results[name].data.gradientNorm:.2g}\t'
            f'{results[name].data.optimizationMessages["Optimization time"]}'
            f'\t{results[name].data.optimizationMessages["Cause of termination"]}'
            f'\n'
        )
    except excep.biogemeError as e:
        print(e)
        results[name] = None
        msg += f'{name}\tFailed to estimate the model'


SUMMARY_FILE = '05normalMixture_allAlgos.log'
with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    print('Algorithm\t\tloglike\t\tnormg\ttime\t\tdiagnostic', file=f)
    print('+++++++++\t\t+++++++\t\t+++++\t++++\t\t++++++++++', file=f)
    print(msg, file=f)
print(f'Summary reported in file {SUMMARY_FILE}')

"""
Here are the results. Note that the draws are identical for all runs. Still, the algorithms
may converge to different solutions. Some algorithms obtain a solution with
B_TIME_S = 1.65 (LL = -5215.588),
and some obtain a solution with
B_TIME_S = -1.65 (LL = -5215.204).
Both are local optima of the likelihood function. As the draws are not exactly symmetric,
these solutions have different values for the objective functions. If the number of draws is
increased, the two local solutions will (asymptotically) become identical.

Algorithm               loglike         normg   time            diagnostic
+++++++++               +++++++         +++++   ++++            ++++++++++
scipy                           -5215.59        0.00024 0:00:26.443081  b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
Line search                     -5215.59        3.8e-06 0:01:39.359296  Relative gradient = 9e-10 <= 6.1e-06
Trust region (dogleg)           -5215.20        0.0066  0:00:27.050893  Relative gradient = 1.5e-06 <= 6.1e-06
Trust region (cg)               -5215.20        0.013   0:00:33.720330  Relative gradient = 2.9e-06 <= 6.1e-06
LS-BFGS                         -5215.59        0.017   0:01:33.475099  Relative gradient = 2.9e-06 <= 6.1e-06
TR-BFGS                         -5215.59        0.018   0:01:22.466036  Relative gradient = 3.1e-06 <= 6.1e-06
Simple bounds Newton fCG        -5215.59        0.026   0:00:30.640365  Relative gradient = 5.1e-06 <= 6.1e-06
Simple bounds BFGS fCG          -5215.59        0.027   0:01:28.154424  Relative gradient = 4.7e-06 <= 6.1e-06
Simple bounds hybrid fCG        -5215.59        0.031   0:00:37.804337  Relative gradient = 3.9e-06 <= 6.1e-06
Simple bounds Newton iCG        -5215.20        0.017   0:00:26.515896  Relative gradient = 4e-06 <= 6.1e-06
Simple bounds BFGS iCG          -5215.59        0.027   0:01:28.327714  Relative gradient = 4.7e-06 <= 6.1e-06
Simple bounds hybrid iCG        -5215.20        0.0044  0:00:32.788981  Relative gradient = 8.9e-07 <= 6.1e-06
"""
