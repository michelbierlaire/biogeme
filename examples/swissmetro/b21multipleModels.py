"""File 21multipleModels.py

:author: Michel Bierlaire, EPFL
:date: Mon Mar 20 08:59:33 2023

 Example of the estimation of several versions of the model using
 assisted specification algorithm Three alternatives: Train, Car and

"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, logzero
from biogeme.results import compileEstimationResults, loglikelihood_dimension
from biogeme.catalog import Catalog, SynchronizedCatalog, segmentation_catalog
from biogeme.segmentation import DiscreteSegmentationTuple
from biogeme.assisted import AssistedSpecification
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
    INCOME,
    GA,
)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Segmentations
gender_segmentation = DiscreteSegmentationTuple(
    variable=MALE,
    mapping={
        0: 'female',
        1: 'male',
    },
)

income_segmentation = DiscreteSegmentationTuple(
    variable=INCOME,
    mapping={
        1: 'inc-under50',
        2: 'inc-50-100',
        3: 'inc-100+',
        4: 'inc-unknown',
    },
)

ga_segmentation = DiscreteSegmentationTuple(variable=GA, mapping={1: 'GA', 0: 'noGA'})

asc_segmentations = (gender_segmentation, ga_segmentation,)
ASC_CAR_catalog = segmentation_catalog(
    beta_parameter=ASC_CAR,
    potential_segmentations=asc_segmentations,
    maximum_number=2,
)
ASC_TRAIN_catalog = segmentation_catalog(
    beta_parameter=ASC_TRAIN,
    potential_segmentations=asc_segmentations,
    maximum_number=2,
    synchronized_with=ASC_CAR_catalog,
)

cost_segmentations = (ga_segmentation, income_segmentation,)
B_COST_catalog = segmentation_catalog(
    beta_parameter=B_COST,
    potential_segmentations=cost_segmentations,
    maximum_number=1,
)

ell_time = Beta('lambda_time', 1, None, None, 0)
# Potential non linear specification of travel time
TRAIN_TT_catalog = Catalog.from_dict(
    catalog_name='TRAIN_TT',
    dict_of_expressions={
        'linear': TRAIN_TT_SCALED,
        'log': logzero(TRAIN_TT_SCALED),
        'boxcox': models.boxcox(TRAIN_TT_SCALED, ell_time),
    },
)

SM_TT_catalog = SynchronizedCatalog.from_dict(
    catalog_name='SM_TT',
    dict_of_expressions={
        'linear': SM_TT_SCALED,
        'log': logzero(SM_TT_SCALED),
        'boxcox': models.boxcox(SM_TT_SCALED, ell_time),
    },
    controller=TRAIN_TT_catalog,
)

CAR_TT_catalog = SynchronizedCatalog.from_dict(
    catalog_name='CAR_TT',
    dict_of_expressions={
        'linear': CAR_TT_SCALED,
        'log': logzero(CAR_TT_SCALED),
        'boxcox': models.boxcox(CAR_TT_SCALED, ell_time),
    },
    controller=TRAIN_TT_catalog,
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

biogeme = bio.BIOGEME(database, logprob)

assisted_specification = AssistedSpecification(
    biogeme_object=biogeme,
    multi_objectives=loglikelihood_dimension,
    pareto_file_name='b21multipleModels.pareto',
)
print('Algorithm info: ')
for m in assisted_specification.statistics():
    print(m)
non_dominated_models = assisted_specification.run()

summary, description = compileEstimationResults(
    non_dominated_models, use_short_names=True
)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')
