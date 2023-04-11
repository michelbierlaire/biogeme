""" File b01logit_all_algos.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:18:01 2023

 Logit model. Estimation with several algorithms.

"""
import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
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

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
# logger.setDetailed()

algos = {
    'scipy                ': 'scipy',
    'Line search          ': 'LS-newton',
    'Trust region (dogleg)': 'TR-newton',
    'Trust region (cg)    ': 'TR-newton',
    'LS-BFGS              ': 'LS-BFGS',
    'TR-BFGS              ': 'TR-BFGS',
    'Simple bounds Newton ': 'simple_bounds',
    'Simple bounds BFGS   ': 'simple_bounds',
    'Simple bounds hybrid ': 'simple_bounds',
}

algoParameters = {
    'Trust region (dogleg)': {'dogleg': True},
    'Trust region (cg)': {'dogleg': False},
    'Simple bounds Newton ': {'second_derivatives': 1.0},
    'Simple bounds BFGS   ': {'second_derivatives': 0.0},
    'Simple bounds hybrid ': {'second_derivatives': 0.5},
}

results = {}
print("Algorithm\t\tloglike\t\tnormg\ttime\t\tdiagnostic")
print("+++++++++\t\t+++++++\t\t+++++\t++++\t\t++++++++++")

for name, algo in algos.items():
    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database, logprob)
    the_biogeme.algorithm_name = algo
    the_biogeme.modelName = f'b01logit_all_algos_{name}'.strip()
    p = algoParameters.get(name)
    if p is not None:
        for attr, value in p.items():
            setattr(the_biogeme, attr, value)
    results[name] = the_biogeme.estimate()
    print(
        f'{name}\t{results[name].data.logLike:.2f}\t'
        f'{results[name].data.gradientNorm:.2g}\t'
        f'{results[name].data.optimizationMessages["Optimization time"]}'
        f'\t{results[name].data.optimizationMessages["Cause of termination"]}'
    )
