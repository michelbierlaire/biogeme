"""File 01logitBis.py

:author: Michel Bierlaire, EPFL
:date: Thu Sep  6 15:14:39 2018

 Example of a logit model.

Same as 01logit, using bioLinearUtility, and introducing some options
 and features.  Three alternatives: Train, Car and Swissmetro SP data

"""
import pandas as pd

import biogeme.biogeme as bio
import biogeme.database as db
from biogeme import models
import biogeme.optimization as opt
import biogeme.messaging as msg
import biogeme.segmentation as seg
from biogeme.expressions import Beta, Variable, bioLinearUtility

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
MALE = Variable('MALE')

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

# We use starting values estimated from a previous run
B_TIME = Beta('B_TIME', -1.28, None, None, 0)
B_COST = Beta('B_COST', -1.08, None, None, 0)

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables by adding columns to the database.
# This is recommended for estimation. And not recommended for simulation.
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

# Define segmentations
gender_segmentation = seg.DiscreteSegmentationTuple(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

GA_segmentation = seg.DiscreteSegmentationTuple(
    variable=GA, mapping={0: 'without_ga', 1: 'with_ga'}
)

segmentations_for_asc = [
    gender_segmentation,
    GA_segmentation,
]

segmented_ASC_TRAIN = seg.segment_parameter(ASC_TRAIN, segmentations_for_asc)
segmented_ASC_CAR = seg.segment_parameter(ASC_CAR, segmentations_for_asc)

# Definition of the utility functions
terms1 = [(B_TIME, TRAIN_TT_SCALED), (B_COST, TRAIN_COST_SCALED)]
V1 = segmented_ASC_TRAIN + bioLinearUtility(terms1)

terms2 = [(B_TIME, SM_TT_SCALED), (B_COST, SM_COST_SCALED)]
V2 = bioLinearUtility(terms2)

terms3 = [(B_TIME, CAR_TT_SCALED), (B_COST, CAR_CO_SCALED)]
V3 = segmented_ASC_CAR + bioLinearUtility(terms3)

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()


# These notes will be included as such in the report file.
user_notes = (
    'Example of a logit model with three alternatives: Train, Car and'
    ' Swissmetro. Same as 01logit and '
    'introducing some options and features. In particular, bioLinearUtility,'
    ' and automatic segmentation of parameters.'
)

# Create the Biogeme object
biogeme = bio.BIOGEME(
    database, logprob, numberOfThreads=2, userNotes=user_notes
)

# As we have used starting values different from 0, the initial model
# is not the equal probability model. If we want to include the latter
# in the results, we need to calculate its log likelihood.
biogeme.calculateNullLoglikelihood(av)

biogeme.modelName = '01logitBis'
biogeme.saveIterations = False

# Estimate the parameters
results = biogeme.estimate(
    bootstrap=100,
    algorithm=opt.bioNewton,
    algoParameters={'maxiter': 1000},
)

biogeme.createLogFile(verbosity=3)

# Get the results in a pandas table
print('Parameters')
print('----------')
pandasResults = results.getEstimatedParameters()
print(pandasResults)

# Get general statistics
print('General statistics')
print('------------------')
stats = results.getGeneralStatistics()
for description, (value, formatting) in stats.items():
    print(f'{description}: {value:{formatting}}')

# Messages from the optimization algorithm
print('Optimization algorithm')
print('----------------------')
for description, message in results.data.optimizationMessages.items():
    print(f'{description}:\t{message}')

# Generate the file in Alogit format
results.writeF12(robustStdErr=True)
results.writeF12(robustStdErr=False)

# Generate LaTeX code with the results
print(results.getLaTeX())
