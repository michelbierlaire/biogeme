"""File 01logitBis.py

:author: Michel Bierlaire, EPFL
:date: Tue Dec  6 15:30:55 2022

 Example of a logit model.

Same as 01logit, using bioLinearUtility, and introducing some options
 and features.  Three alternatives: Train, Car and Swissmetro SP data

Note that the parameters are defined in the file 01logitBis.toml
"""

import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
import biogeme.segmentation as seg
from biogeme.expressions import Beta, bioLinearUtility

from swissmetro import (
    database,
    CHOICE,
    GA,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    MALE,
    SM_AV,
)


# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)

# We use starting values estimated from a previous run
B_TIME = Beta('B_TIME', -1.28, None, None, 0)
B_COST = Beta('B_COST', -1.08, None, None, 0)

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

ASC_TRAIN_segmentation = seg.Segmentation(ASC_TRAIN, segmentations_for_asc)
segmented_ASC_TRAIN = ASC_TRAIN_segmentation.segmented_beta()
ASC_CAR_segmentation = seg.Segmentation(ASC_CAR, segmentations_for_asc)
segmented_ASC_CAR = ASC_CAR_segmentation.segmented_beta()

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
USER_NOTES = (
    'Example of a logit model with three alternatives: Train, Car and'
    ' Swissmetro. Same as 01logit and '
    'introducing some options and features. In particular, bioLinearUtility,'
    ' and automatic segmentation of parameters.'
)

# Create the Biogeme object
biogeme = bio.BIOGEME(
    database, logprob, userNotes=USER_NOTES, parameter_file='01logitBis.toml'
)

# As we have used starting values different from 0, the initial model
# is not the equal probability model. If we want to include the latter
# in the results, we need to calculate its log likelihood.
biogeme.calculateNullLoglikelihood(av)

biogeme.modelName = '01logitBis'
biogeme.saveIterations = False

# Estimate the parameters
results = biogeme.estimate(bootstrap=100)

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
