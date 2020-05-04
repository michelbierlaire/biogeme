""" File 01logit_allAlgos.py

:author: Michel Bierlaire, EPFL
:date: Sat Sep  7 17:57:16 2019

 Logit model
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import biogeme.biogeme as bio
import biogeme.optimization as opt
import biogeme.models as models
import biogeme.messaging as msg

# The utility functions are defined in a separate file and used for all the examples.
# It allows to avoid duplicate code.
from utilSwissmetro import V, av, database

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
#logger.setWarning()
#logger.setGeneral()
#logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)

algos = {'CFSQP                ': None,
         'scipy                ': opt.scipy,
         'Line search          ': opt.newtonLineSearchForBiogeme,
         'Trust region (dogleg)': opt.newtonTrustRegionForBiogeme,
         'Trust region (cg)    ': opt.newtonTrustRegionForBiogeme,
         'LS-BFGS              ': opt.bfgsLineSearchForBiogeme,
         'TR-BFGS              ': opt.bfgsTrustRegionForBiogeme,
         'Simple bounds Newton ': opt.simpleBoundsNewtonAlgorithmForBiogeme,
         'Simple bounds BFGS   ': opt.simpleBoundsNewtonAlgorithmForBiogeme,
         'Simple bounds hybrid ': opt.simpleBoundsNewtonAlgorithmForBiogeme}

algoParameters = {'Trust region (dogleg)': {'dogleg':True},
                  'Trust region (cg)': {'dogleg':False},
                  'Simple bounds Newton': {'proportionAnalyticalHessian': 1.0},
                  'Simple bounds BFGS': {'proportionAnalyticalHessian': 0.0},
                  'Simple bounds hybrid': {'proportionAnalyticalHessian': 0.5}}

results = {}
print("Algorithm\t\tloglike\t\tnormg\ttime\t\tdiagnostic")
print("+++++++++\t\t+++++++\t\t+++++\t++++\t\t++++++++++")

for name, algo in algos.items():
    biogeme.modelName = f'01logit_allAlgos_{name}'.strip()
    p = algoParameters.get(name)
    results[name] = biogeme.estimate(algorithm=algo, algoParameters=p)
    g = results[name].data.g
    print(f'{name}\t{results[name].data.logLike:.2f}\t'
          f'{results[name].data.gradientNorm:.2g}\t'
          f'{results[name].data.optimizationMessages["Optimization time"]}'
          f'\t{results[name].data.optimizationMessages["Cause of termination"]}')
