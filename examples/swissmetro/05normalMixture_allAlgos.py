"""File 05normalMixture_allAlgos.py

:author: Michel Bierlaire, EPFL
:date: Fri May  1 11:59:20 2020

 Example of a mixture of logit models, using Monte-Carlo integration.
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import pandas as pd
import biogeme.optimization as opt
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
import biogeme.exceptions as excep
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to investigate the database. For example:
# print(database.data.describe())

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

# Removing some observations can be done directly using pandas.
# remove = (((database.data.PURPOSE != 1) &
#           (database.data.PURPOSE != 3)) |
#          (database.data.CHOICE == 0))
# database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

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

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
TRAIN_COST_SCALED = database.DefineVariable(
    'TRAIN_COST_SCALED', TRAIN_COST / 100
)
SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', SM_TT / 100.0)
SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)

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

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=100000)

algos = {
    'scipy                   ': opt.scipy,
    'Line search             ': opt.newtonLineSearchForBiogeme,
    'Trust region (dogleg)   ': opt.newtonTrustRegionForBiogeme,
    'Trust region (cg)       ': opt.newtonTrustRegionForBiogeme,
    'LS-BFGS                 ': opt.bfgsLineSearchForBiogeme,
    'TR-BFGS                 ': opt.bfgsTrustRegionForBiogeme,
    'Simple bounds Newton fCG': opt.simpleBoundsNewtonAlgorithmForBiogeme,
    'Simple bounds BFGS fCG  ': opt.simpleBoundsNewtonAlgorithmForBiogeme,
    'Simple bounds hybrid fCG': opt.simpleBoundsNewtonAlgorithmForBiogeme,
    'Simple bounds Newton iCG': opt.simpleBoundsNewtonAlgorithmForBiogeme,
    'Simple bounds BFGS iCG  ': opt.simpleBoundsNewtonAlgorithmForBiogeme,
    'Simple bounds hybrid iCG': opt.simpleBoundsNewtonAlgorithmForBiogeme,
}

algoParameters = {
    'Trust region (dogleg)   ': {'dogleg': True},
    'Trust region (cg)       ': {'dogleg': False},
    'Simple bounds Newton fCG': {
        'proportionAnalyticalHessian': 1.0,
        'infeasibleConjugateGradient': False,
    },
    'Simple bounds BFGS fCG  ': {
        'proportionAnalyticalHessian': 0.0,
        'infeasibleConjugateGradient': False,
    },
    'Simple bounds hybrid fCG': {
        'proportionAnalyticalHessian': 0.5,
        'infeasibleConjugateGradient': False,
    },
    'Simple bounds Newton iCG': {
        'proportionAnalyticalHessian': 1.0,
        'infeasibleConjugateGradient': True,
    },
    'Simple bounds BFGS iCG  ': {
        'proportionAnalyticalHessian': 0.0,
        'infeasibleConjugateGradient': True,
    },
    'Simple bounds hybrid iCG': {
        'proportionAnalyticalHessian': 0.5,
        'infeasibleConjugateGradient': True,
    },
}

results = {}
msg = ''
for name, algo in algos.items():
    biogeme.modelName = f'05normalMixture_allAlgos_{name}'.strip()
    p = algoParameters.get(name)
    try:
        results[name] = biogeme.estimate(algorithm=algo, algoParameters=p)
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


print('Algorithm\t\tloglike\t\tnormg\ttime\t\tdiagnostic')
print('+++++++++\t\t+++++++\t\t+++++\t++++\t\t++++++++++')
print(msg)

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
