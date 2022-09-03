"""File 13panel_simul.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 16:25:16 2021

 Example of a mixture of logit models, using Monte-Carlo integration.
 The datafile is organized as panel data.
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import sys
import pickle
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.exceptions as excep
import biogeme.results as res
import biogeme.messaging as msg
from biogeme.expressions import (
    Beta,
    Variable,
    bioDraws,
    PanelLikelihoodTrajectory,
    MonteCarlo,
    log,
)


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

# They are organized as panel data. The variable ID identifies each individual.
database.panel("ID")


# Parameters to be estimated
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation
B_TIME = Beta('B_TIME', 0, None, None, 0)

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL_ANTI')

# We do the same for the constants, to address serial correlation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR_S = Beta('ASC_CAR_S', 1, None, None, 0)
ASC_CAR_RND = ASC_CAR + ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL_ANTI')

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_TRAIN_S = Beta('ASC_TRAIN_S', 1, None, None, 0)
ASC_TRAIN_RND = ASC_TRAIN + ASC_TRAIN_S * bioDraws(
    'ASC_TRAIN_RND', 'NORMAL_ANTI'
)

ASC_SM = Beta('ASC_SM', 0, None, None, 1)
ASC_SM_S = Beta('ASC_SM_S', 1, None, None, 0)
ASC_SM_RND = ASC_SM + ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL_ANTI')

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
V1 = ASC_TRAIN_RND + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM_RND + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR_RND + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Conditional to the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel)
obsprob = models.logit(V, av, CHOICE)

# Conditional to the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
condprobIndiv = PanelLikelihoodTrajectory(obsprob)

# We integrate over the random parameters using Monte-Carlo
logprob = log(MonteCarlo(condprobIndiv))

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
logger.setDetailed()
# logger.setDebug()

# Estimate the parameters.
try:
    results = res.bioResults(pickleFile='12panel.pickle')
except excep.biogemeError:
    sys.exit(
        'Run first the script 12panel.py '
        'in order to generate the '
        'file 12panel.pickle.'
    )

# Simulate to recalculate the log likelihood
simulated_loglike = logprob.getValue_c(
    database=database,
    betas=results.getBetaValues(),
    numberOfDraws=100000,
    aggregation=True,
    prepareIds=True,
)

print(f'Simulated log likelihood: {simulated_loglike}')

numerator = MonteCarlo(B_TIME_RND * condprobIndiv)
denominator = MonteCarlo(condprobIndiv)
simulate = {
    'Numerator': numerator,
    'Denominator': denominator,
}

biosim = bio.BIOGEME(
    database, simulate, numberOfDraws=100000, suggestScales=False
)
sim = biosim.simulate()

sim['Individual-level parameters'] = sim['Numerator'] / sim['Denominator']
pickleFile = '13panel_individual_parameters.pickle'
with open(pickleFile, 'wb') as f:
    pickle.dump(sim, f)

htmlFile = '13panel_simul.html'
with open(htmlFile, 'w') as h:
    print(sim.to_html(), file=h)
print(f'Simulated values available in {htmlFile}')
