"""File 22multipleModels_bis.py

:author: Michel Bierlaire, EPFL
:date: Fri Apr  7 18:25:46 2023

Same example as b22multipleModels, where we enumerate explicitly all specifications

"""
import biogeme.logging as blog

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, logzero
from biogeme.results import compileEstimationResults, AIC_BIC_dimension
from biogeme.catalog import Catalog, SynchronizedCatalog
from biogeme.assisted import AssistedSpecification
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

ASC_CAR_catalog = Catalog.from_dict(
    'ASC_CAR_catalog',
    {
        'no_segment': ASC_CAR,
        'gender_segment': ASC_CAR_gender_segmentation.segmented_beta(),
        'luggage_segment': ASC_CAR_luggage_segmentation.segmented_beta(),
        'GA_segment': ASC_CAR_GA_segmentation.segmented_beta(),
    },
)

ASC_TRAIN_catalog = SynchronizedCatalog.from_dict(
    'ASC_TRAIN_catalog',
    {
        'no_segment': ASC_TRAIN,
        'gender_segment': ASC_TRAIN_gender_segmentation.segmented_beta(),
        'luggage_segment': ASC_TRAIN_luggage_segmentation.segmented_beta(),
        'GA_segment': ASC_TRAIN_GA_segmentation.segmented_beta(),
    },
    ASC_CAR_catalog,
)

# We define a catalog with two different specifications for headway
TRAIN_HEADWAY_catalog = Catalog.from_dict(
    'TRAIN_HEADWAY_catalog',
    {'without_headway': 0, 'with_headway': B_HEADWAY * TRAIN_HE},
)

SM_HEADWAY_catalog = SynchronizedCatalog.from_dict(
    'SM_HEADWAY_catalog',
    {'without_headway': 0, 'with_headway': B_HEADWAY * SM_HE},
    TRAIN_HEADWAY_catalog,
)


ell_TT = Beta('lambda_TT', 1, None, None, 0)

TRAIN_TT_catalog = Catalog.from_dict(
    'TRAIN_TT_catalog',
    {
        'linear': TRAIN_TT_SCALED,
        'log': logzero(TRAIN_TT_SCALED),
        'sqrt': TRAIN_TT_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(TRAIN_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(TRAIN_TT_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(TRAIN_TT_SCALED, ell_TT),
    },
)

SM_TT_catalog = SynchronizedCatalog.from_dict(
    'SM_TT_catalog',
    {
        'linear': SM_TT_SCALED,
        'log': logzero(SM_TT_SCALED),
        'sqrt': SM_TT_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(SM_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(SM_TT_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(SM_TT_SCALED, ell_TT),
    },
    TRAIN_TT_catalog,
)

CAR_TT_catalog = SynchronizedCatalog.from_dict(
    'CAR_TT_catalog',
    {
        'linear': CAR_TT_SCALED,
        'log': logzero(CAR_TT_SCALED),
        'sqrt': CAR_TT_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(CAR_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(CAR_TT_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(CAR_TT_SCALED, ell_TT),
    },
    TRAIN_TT_catalog,
)

ell_COST = Beta('lambda_COST', 1, None, None, 0)

TRAIN_COST_catalog = Catalog.from_dict(
    'TRAIN_COST_catalog',
    {
        'linear': TRAIN_COST_SCALED,
        'log': logzero(TRAIN_COST_SCALED),
        'sqrt': TRAIN_COST_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(TRAIN_COST_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(TRAIN_COST_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(TRAIN_COST_SCALED, ell_COST),
    },
)

SM_COST_catalog = SynchronizedCatalog.from_dict(
    'SM_COST_catalog',
    {
        'linear': SM_COST_SCALED,
        'log': logzero(SM_COST_SCALED),
        'sqrt': SM_COST_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(SM_COST_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(SM_COST_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(SM_COST_SCALED, ell_COST),
    },
    TRAIN_COST_catalog,
)

CAR_COST_catalog = SynchronizedCatalog.from_dict(
    'CAR_COST_catalog',
    {
        'linear': CAR_CO_SCALED,
        'log': logzero(CAR_CO_SCALED),
        'sqrt': CAR_CO_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(CAR_CO_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(CAR_CO_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(CAR_CO_SCALED, ell_COST),
    },
    TRAIN_COST_catalog,
)


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

# We now consider a model with the log travel time instead of the cost

PARETO_FILE_NAME = 'b22multipleModels_bis.pareto'

the_biogeme = bio.BIOGEME(database, logprob)
# We increase the maximum number of specification, to force the
# algorithm to enumerate all of them
the_biogeme.maximum_number_catalog_expressions = 300


nbr = logprob.number_of_multiple_expressions()
print(f'There a {nbr} possible specifications')
assisted_specification = AssistedSpecification(
    biogeme_object=the_biogeme,
    multi_objectives=AIC_BIC_dimension,
    pareto_file_name=PARETO_FILE_NAME,
    max_neighborhood=30,
)
print(assisted_specification.statistics())
non_dominated_models = assisted_specification.run(
    number_of_neighbors=30,
)

summary, description = compileEstimationResults(
    non_dominated_models, use_short_names=True
)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')

assisted_specification.plot()
