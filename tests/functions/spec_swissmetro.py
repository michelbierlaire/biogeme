"""Specification of a multiple expressions used for testing purposes

"""

from biogeme.expressions import Beta
from biogeme import models
import biogeme.segmentation as seg
from biogeme.catalog import segmentation_catalogs

from swissmetro_data import (
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
)

ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

gender_segmentation = seg.DiscreteSegmentationTuple(
    variable=MALE,
    mapping={
        0: 'female',
        1: 'male',
    },
)


ASC_CAR_catalog, ASC_TRAIN_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_CAR, ASC_TRAIN],
    potential_segmentations=(gender_segmentation,),
    maximum_number=1,
)

# Definition of the utility functions with linear cost
V1 = ASC_TRAIN_catalog + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR_catalog + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

logprob = models.loglogit(V, av, CHOICE)
