""" File 01logit_allAlgos.py

:author: Michel Bierlaire, EPFL
:date: Sat Sep  7 17:57:16 2019

 Logit model
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import pandas as pd
import biogeme.biogeme as bio
import biogeme.optimization as opt
import biogeme.database as db
from biogeme import models
import biogeme.messaging as msg
from biogeme.expressions import Beta, Variable

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

PURPOSE = Variable('PURPOSE')
CHOICE = Variable('CHOICE')
GA = Variable('GA')
TRAIN_CO = Variable('TRAIN_CO')
CAR_AV = Variable('CAR_AV')
SP = Variable('SP')
TRAIN_AV = Variable('TRAIN_AV')
TRAIN_TT = Variable('TRAIN_TT')
SM_TT = Variable('SM_TT')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')
SM_CO = Variable('SM_CO')
SM_AV = Variable('SM_AV')


# Removing some observations
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

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

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)

algos = {
    'scipy                ': opt.scipy,
    'Line search          ': opt.newtonLineSearchForBiogeme,
    'Trust region (dogleg)': opt.newtonTrustRegionForBiogeme,
    'Trust region (cg)    ': opt.newtonTrustRegionForBiogeme,
    'LS-BFGS              ': opt.bfgsLineSearchForBiogeme,
    'TR-BFGS              ': opt.bfgsTrustRegionForBiogeme,
    'Simple bounds Newton ': opt.simpleBoundsNewtonAlgorithmForBiogeme,
    'Simple bounds BFGS   ': opt.simpleBoundsNewtonAlgorithmForBiogeme,
    'Simple bounds hybrid ': opt.simpleBoundsNewtonAlgorithmForBiogeme,
}

algoParameters = {
    'Trust region (dogleg)': {'dogleg': True},
    'Trust region (cg)': {'dogleg': False},
    'Simple bounds Newton ': {'proportionAnalyticalHessian': 1.0},
    'Simple bounds BFGS   ': {'proportionAnalyticalHessian': 0.0},
    'Simple bounds hybrid ': {'proportionAnalyticalHessian': 0.5},
}

results = {}
print("Algorithm\t\tloglike\t\tnormg\ttime\t\tdiagnostic")
print("+++++++++\t\t+++++++\t\t+++++\t++++\t\t++++++++++")

for name, algo in algos.items():
    biogeme.modelName = f'01logit_allAlgos_{name}'.strip()
    p = algoParameters.get(name)
    results[name] = biogeme.estimate(algorithm=algo, algoParameters=p)
    print(
        f'{name}\t{results[name].data.logLike:.2f}\t'
        f'{results[name].data.gradientNorm:.2g}\t'
        f'{results[name].data.optimizationMessages["Optimization time"]}'
        f'\t{results[name].data.optimizationMessages["Cause of termination"]}'
    )
