"""File 09nested.py

:author: Michel Bierlaire, EPFL
:date: Tue Dec  6 18:24:29 2022

 Example of a nested logit model.
 Three alternatives: Train, Car and Swissmetro
 Train and car are in the same nest.
 SP data
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
from biogeme.expressions import Beta, Variable

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
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
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
MU = Beta('MU', 1, 1, 10, 0)

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

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()


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
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = f'09nested_allAlgos_{name}'.strip()
    biogeme.algorithm_name = algo
    p = algoParameters.get(name)
    if p is not None:
        for attr, value in p.items():
            setattr(biogeme, attr, value)
    results[name] = biogeme.estimate()
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

"""
Here are the results.

Algorithm			loglike		normg	time		diagnostic
+++++++++			+++++++		+++++	++++		++++++++++
scipy                   	-5236.90	0.00017	0:00:00.388892	CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
Simple bounds Newton    	-5236.90	0.0027	0:00:00.221994	Relative gradient = 3.5e-07 <= 6.1e-06
Simple bounds BFGS      	-5236.90	0.029	0:00:00.923152	Relative gradient = 5.1e-06 <= 6.1e-06
Simple bounds hybrid 20%	-5236.90	0.024	0:00:00.495556	Relative gradient = 2.7e-06 <= 6.1e-06
Simple bounds hybrid 50%	-5236.90	0.0074	0:00:00.374580	Relative gradient = 1.4e-06 <= 6.1e-06
Simple bounds hybrid 80%	-5236.90	0.0096	0:00:00.336953	Relative gradient = 1.9e-06 <= 6.1e-06

"""
