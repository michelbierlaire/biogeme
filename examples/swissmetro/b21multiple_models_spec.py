"""File b21multiple_models_spec.py

:author: Michel Bierlaire, EPFL
:date: Fri Jul 21 17:46:09 2023

 Example of the estimation of several versions of the model using
 assisted specification algorithm. Specification of the catalogs.

"""
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, logzero
from biogeme.catalog import Catalog, segmentation_catalogs

from swissmetro_data import (
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
    INCOME,
    GA,
)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Segmentations
gender_segmentation = database.generate_segmentation(
    variable=MALE,
    mapping={
        0: 'female',
        1: 'male',
    },
)

income_segmentation = database.generate_segmentation(
    variable=INCOME,
    mapping={
        0: 'inc-zero',
        1: 'inc-under50',
        2: 'inc-50-100',
        3: 'inc-100+',
        4: 'inc-unknown',
    },
)

print(f'{income_segmentation=}')
ga_segmentation = database.generate_segmentation(
    variable=GA, mapping={1: 'GA', 0: 'noGA'}
)

asc_segmentations = (
    gender_segmentation,
    ga_segmentation,
)

ASC_CAR_catalog, ASC_TRAIN_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_CAR, ASC_TRAIN],
    potential_segmentations=asc_segmentations,
    maximum_number=2,
)

cost_segmentations = (
    ga_segmentation,
    income_segmentation,
)

# Note that the function returns a list. In this case, it contains
# only one element. This is the reason of the presence of the comma
# after B_COST_catalog
(B_COST_catalog,) = segmentation_catalogs(
    generic_name='B_COST',
    beta_parameters=[B_COST],
    potential_segmentations=cost_segmentations,
    maximum_number=1,
)

ell_time = Beta('lambda_time', 1, None, 10, 0)
# Potential non linear specification of travel time
TRAIN_TT_catalog = Catalog.from_dict(
    catalog_name='TRAIN_TT',
    dict_of_expressions={
        'linear': TRAIN_TT_SCALED,
        'log': logzero(TRAIN_TT_SCALED),
        'boxcox': models.boxcox(TRAIN_TT_SCALED, ell_time),
    },
)

SM_TT_catalog = Catalog.from_dict(
    catalog_name='SM_TT',
    dict_of_expressions={
        'linear': SM_TT_SCALED,
        'log': logzero(SM_TT_SCALED),
        'boxcox': models.boxcox(SM_TT_SCALED, ell_time),
    },
    controlled_by=TRAIN_TT_catalog.controlled_by,
)

CAR_TT_catalog = Catalog.from_dict(
    catalog_name='CAR_TT',
    dict_of_expressions={
        'linear': CAR_TT_SCALED,
        'log': logzero(CAR_TT_SCALED),
        'boxcox': models.boxcox(CAR_TT_SCALED, ell_time),
    },
    controlled_by=TRAIN_TT_catalog.controlled_by,
)

# Definition of the utility functions with linear cost
V1 = ASC_TRAIN_catalog + B_TIME * TRAIN_TT_catalog + B_COST_catalog * TRAIN_COST_SCALED
V2 = B_TIME * SM_TT_catalog + B_COST_catalog * SM_COST_SCALED
V3 = ASC_CAR_catalog + B_TIME * CAR_TT_catalog + B_COST_catalog * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

print(
    f'Total number of possible specifications: '
    f'{logprob.number_of_multiple_expressions()}'
)

the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b21multiple_models'

PARETO_FILE_NAME = 'b21multiple_models.pareto'
