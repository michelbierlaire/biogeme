"""File 11cnl_simul.py

:author: Michel Bierlaire, EPFL
:date: Sun Sep  8 11:13:22 2019

 Example of simulation with a cross-nested logit model.
 Three alternatives: Train, Car and Swissmetro
 Train and car are in the same nest.
 SP data
"""

import sys
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.tools import calculate_correlation
import biogeme.results as res
from biogeme.expressions import Beta, Variable, Derive
import biogeme.exceptions as excep

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

# Simulation should be done with estimated value of the
# parameters. You can include them manually. Here, we prefer to set
# them to zero, and read the values from the file created after
# estimation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

MU_EXISTING = Beta('MU_EXISTING', 1, 1, None, 0)
MU_PUBLIC = Beta('MU_PUBLIC', 1, 1, None, 0)
ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.5, 0, 1, 0)
ALPHA_PUBLIC = 1 - ALPHA_EXISTING

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: in simulation, do not use the
# DefineVariable operator, as it hides the functional
# relationships. In particular, derivatives cannot be calculated.
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
# Definition of nests
# Nest membership parameters
alpha_existing = {1: ALPHA_EXISTING, 2: 0.0, 3: 1.0}

alpha_public = {1: ALPHA_PUBLIC, 2: 1.0, 3: 0.0}

nest_existing = MU_EXISTING, alpha_existing
nest_public = MU_PUBLIC, alpha_public
nests = nest_existing, nest_public

# Instead of estimating the parameters, read the estimation
# results from the pickle file.
try:
    results = res.bioResults(pickleFile='11cnl.pickle')
except excep.biogemeError:
    print(
        'Run first the script 11cnl.py in order to generate the file '
        '11cnl.pickle.'
    )
    sys.exit()

print('Estimation results: ', results.getEstimatedParameters())

print(
    'Calculating correlation matrix. '
    'It may generate numerical warnings from scipy.'
)
corr = calculate_correlation(
    nests, results, alternative_names={1: 'Train', 2: 'Swissmetro', 3: 'Car'}
)
print(corr)

# The choice model is a cross-nested logit, with availability conditions
prob1 = models.cnl_avail(V, av, nests, 1)
prob2 = models.cnl_avail(V, av, nests, 2)
prob3 = models.cnl_avail(V, av, nests, 3)

# We calculate elasticities
genelas1 = Derive(prob1, 'TRAIN_TT') * TRAIN_TT / prob1
genelas2 = Derive(prob2, 'SM_TT') * SM_TT / prob2
genelas3 = Derive(prob3, 'CAR_TT') * CAR_TT / prob3

# We report the probability of each alternative and the elasticities
simulate = {
    'Prob. train': prob1,
    'Prob. Swissmetro': prob2,
    'Prob. car': prob3,
    'Elas. 1': genelas1,
    'Elas. 2': genelas2,
    'Elas. 3': genelas3,
}

# Create the Biogeme object
biosim = bio.BIOGEME(database, simulate)
biosim.modelName = '11cnl_simul'

# Perform the simulation
simresults = biosim.simulate(results.getBetaValues())
print('Simulation results')
print(simresults.describe())

print(f'Aggregate share of train: {100*simresults["Prob. train"].mean():.1f}%')
