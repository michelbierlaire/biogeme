"""File b22_model1.py

:author: Michel Bierlaire, EPFL
:date: Fri Apr  7 18:41:23 2023

The script b22multipleModels.py has identified 5 Pareto optimal
specifications of the models.  In the files b22_modelk.py, we
reestimate those models without relying on Catalogs, in order to show
that it is equivalent.

Model_1:
  ASC_CAR_catalog:no_segment
  TRAIN_COST_catalog:sqrt
  TRAIN_HEADWAY_catalog:with_headway
  TRAIN_TT_catalog:sqrt

"""
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from biogeme.segmentation import DiscreteSegmentationTuple, Segmentation
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
    MALE,
    TRAIN_HE,
    SM_HE,
    LUGGAGE,
    GA,
)


# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
B_HEADWAY = Beta('B_HEADWAY', 0, None, None, 0)

# Define segmentations
gender_segmentation = DiscreteSegmentationTuple(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

GA_segmentation = DiscreteSegmentationTuple(
    variable=GA, mapping={0: 'without_ga', 1: 'with_ga'}
)

luggage_segmentation = DiscreteSegmentationTuple(
    variable=LUGGAGE, mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'}
)

ASC_CAR_gender_segmentation = Segmentation(ASC_CAR, [gender_segmentation])
ASC_TRAIN_gender_segmentation = Segmentation(ASC_TRAIN, [gender_segmentation])

ASC_CAR_luggage_segmentation = Segmentation(ASC_CAR, [luggage_segmentation])
ASC_TRAIN_luggage_segmentation = Segmentation(ASC_TRAIN, [luggage_segmentation])

ASC_CAR_GA_segmentation = Segmentation(ASC_CAR, [GA_segmentation])
ASC_TRAIN_GA_segmentation = Segmentation(ASC_TRAIN, [GA_segmentation])

ASC_CAR_catalog = ASC_CAR
ASC_TRAIN_catalog = ASC_TRAIN

TRAIN_HEADWAY_catalog = B_HEADWAY * TRAIN_HE
SM_HEADWAY_catalog = B_HEADWAY * SM_HE
TRAIN_TT_catalog = TRAIN_TT_SCALED**0.5
SM_TT_catalog = SM_TT_SCALED**0.5
CAR_TT_catalog = CAR_TT_SCALED**0.5
TRAIN_COST_catalog = TRAIN_COST_SCALED**0.5
SM_COST_catalog = SM_COST_SCALED**0.5
CAR_COST_catalog = CAR_CO_SCALED**0.5

# Definition of the utility functions with linear cost
V1 = (
    ASC_TRAIN_catalog
    + B_TIME * TRAIN_TT_catalog
    + B_COST * TRAIN_COST_catalog
    + TRAIN_HEADWAY_catalog
)
V2 = B_TIME * SM_TT_catalog + B_COST * SM_COST_catalog + SM_HEADWAY_catalog
V3 = ASC_CAR_catalog + B_TIME * CAR_TT_catalog + B_COST * CAR_COST_catalog

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b22_model1'
the_biogeme.calculateNullLoglikelihood(av)

# Estimate the parameters
results = the_biogeme.estimate()

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
