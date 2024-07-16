""".. _plot_b22multiple_models_spec:

Specification of a catalog of models
====================================

Specification of the Catalog of expressions for the assisted
specification algorithm. Note that this script does not perform any
estimation. It is imported by other scripts:
:ref:`plot_b22multiple_models`, :ref:`plot_b22process_pareto`.

:author: Michel Bierlaire, EPFL
:date: Fri Jul 21 17:56:47 2023

"""
from biogeme import models
import biogeme.biogeme as bio
from biogeme.expressions import Beta, logzero
from biogeme.catalog import Catalog, segmentation_catalogs

# %%
# See the data processing script: :ref:`swissmetro_data`.
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
    TRAIN_HE,
    SM_HE,
    LUGGAGE,
    GA,
)

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
B_HEADWAY = Beta('B_HEADWAY', 0, None, None, 0)

# %%
# Define segmentations
gender_segmentation = database.generate_segmentation(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

GA_segmentation = database.generate_segmentation(
    variable=GA, mapping={0: 'without_ga', 1: 'with_ga'}
)

luggage_segmentation = database.generate_segmentation(
    variable=LUGGAGE, mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'}
)

ASC_CAR_catalog, ASC_TRAIN_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_CAR, ASC_TRAIN],
    potential_segmentations=[
        gender_segmentation,
        luggage_segmentation,
        GA_segmentation,
    ],
    maximum_number=2,
)

# %%
# We define a catalog with two different specifications for headway.
TRAIN_HEADWAY_catalog = Catalog.from_dict(
    catalog_name='TRAIN_HEADWAY_catalog',
    dict_of_expressions={'without_headway': 0, 'with_headway': B_HEADWAY * TRAIN_HE},
)

SM_HEADWAY_catalog = Catalog.from_dict(
    catalog_name='SM_HEADWAY_catalog',
    dict_of_expressions={'without_headway': 0, 'with_headway': B_HEADWAY * SM_HE},
    controlled_by=TRAIN_HEADWAY_catalog.controlled_by,
)


# %%
# Parameter for Box-Cox transforms.
ell_TT = Beta('lambda_TT', 1, None, 10, 0)

# %%
# Non linear specification for travel time.
TRAIN_TT_catalog = Catalog.from_dict(
    catalog_name='TRAIN_TT_catalog',
    dict_of_expressions={
        'linear': TRAIN_TT_SCALED,
        'log': logzero(TRAIN_TT_SCALED),
        'sqrt': TRAIN_TT_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(TRAIN_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(TRAIN_TT_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(TRAIN_TT_SCALED, ell_TT),
    },
)

SM_TT_catalog = Catalog.from_dict(
    catalog_name='SM_TT_catalog',
    dict_of_expressions={
        'linear': SM_TT_SCALED,
        'log': logzero(SM_TT_SCALED),
        'sqrt': SM_TT_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(SM_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(SM_TT_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(SM_TT_SCALED, ell_TT),
    },
    controlled_by=TRAIN_TT_catalog.controlled_by,
)

CAR_TT_catalog = Catalog.from_dict(
    catalog_name='CAR_TT_catalog',
    dict_of_expressions={
        'linear': CAR_TT_SCALED,
        'log': logzero(CAR_TT_SCALED),
        'sqrt': CAR_TT_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(CAR_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(CAR_TT_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(CAR_TT_SCALED, ell_TT),
    },
    controlled_by=TRAIN_TT_catalog.controlled_by,
)

# %%
# Parameter for Box-Cox transforms.
ell_COST = Beta('lambda_COST', 1, None, 10, 0)

# %%
# Nonlinear transformations for travel cost.
TRAIN_COST_catalog = Catalog.from_dict(
    catalog_name='TRAIN_COST_catalog',
    dict_of_expressions={
        'linear': TRAIN_COST_SCALED,
        'log': logzero(TRAIN_COST_SCALED),
        'sqrt': TRAIN_COST_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(TRAIN_COST_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(TRAIN_COST_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(TRAIN_COST_SCALED, ell_COST),
    },
)

SM_COST_catalog = Catalog.from_dict(
    catalog_name='SM_COST_catalog',
    dict_of_expressions={
        'linear': SM_COST_SCALED,
        'log': logzero(SM_COST_SCALED),
        'sqrt': SM_COST_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(SM_COST_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(SM_COST_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(SM_COST_SCALED, ell_COST),
    },
    controlled_by=TRAIN_COST_catalog.controlled_by,
)

CAR_COST_catalog = Catalog.from_dict(
    catalog_name='CAR_COST_catalog',
    dict_of_expressions={
        'linear': CAR_CO_SCALED,
        'log': logzero(CAR_CO_SCALED),
        'sqrt': CAR_CO_SCALED**0.5,
        'piecewise_1': models.piecewiseFormula(CAR_CO_SCALED, [0, 0.1, None]),
        'piecewise_2': models.piecewiseFormula(CAR_CO_SCALED, [0, 0.25, None]),
        'boxcox': models.boxcox(CAR_CO_SCALED, ell_COST),
    },
    controlled_by=TRAIN_COST_catalog.controlled_by,
)

# %%
# Definition of the utility functions
V1 = (
    ASC_TRAIN_catalog
    + B_TIME * TRAIN_TT_catalog
    + B_COST * TRAIN_COST_catalog
    + TRAIN_HEADWAY_catalog
)
V2 = B_TIME * SM_TT_catalog + B_COST * SM_COST_catalog + SM_HEADWAY_catalog

V3 = ASC_CAR_catalog + B_TIME * CAR_TT_catalog + B_COST * CAR_COST_catalog

# %%
# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# %%
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b22multiple_models'

# %%
PARETO_FILE_NAME = 'b22multiple_models.pareto'
