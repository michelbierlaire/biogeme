""".. _plot_b22multiple_models_spec:

Specification of a catalog of models
====================================

Specification of the Catalog of expressions for the assisted
specification algorithm. Note that this script does not perform any
estimation. It is imported by other scripts:
:ref:`plot_b22multiple_models`, :ref:`plot_b22process_pareto`.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 12:32:54
"""

from biogeme.biogeme import BIOGEME
from biogeme.catalog import Catalog, segmentation_catalogs
from biogeme.expressions import Beta, logzero
from biogeme.models import boxcox, loglogit, piecewise_formula

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    LUGGAGE,
    MALE,
    SM_AV,
    SM_COST_SCALED,
    SM_HE,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_HE,
    TRAIN_TT_SCALED,
    database,
)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)
b_headway = Beta('b_headway', 0, None, None, 0)

# %%
# Define segmentations
gender_segmentation = database.generate_segmentation(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

ga_segmentation = database.generate_segmentation(
    variable=GA, mapping={0: 'without_ga', 1: 'with_ga'}
)

luggage_segmentation = database.generate_segmentation(
    variable=LUGGAGE, mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'}
)

asc_car_catalog, asc_train_catalog = segmentation_catalogs(
    generic_name='asc',
    beta_parameters=[asc_car, asc_train],
    potential_segmentations=(
        gender_segmentation,
        luggage_segmentation,
        ga_segmentation,
    ),
    maximum_number=2,
)

# %%
# We define a catalog with two different specifications for headway.
train_headway_catalog = Catalog.from_dict(
    catalog_name='train_headway_catalog',
    dict_of_expressions={'without_headway': 0, 'with_headway': b_headway * TRAIN_HE},
)

sm_headway_catalog = Catalog.from_dict(
    catalog_name='sm_headway_catalog',
    dict_of_expressions={'without_headway': 0, 'with_headway': b_headway * SM_HE},
    controlled_by=train_headway_catalog.controlled_by,
)


# %%
# Parameter for Box-Cox transforms.
ell_tt = Beta('lambda_tt', 1, -10, 10, 0)

# %%
# Non-linear specification for travel time.
train_tt_catalog = Catalog.from_dict(
    catalog_name='train_tt_catalog',
    dict_of_expressions={
        'linear': TRAIN_TT_SCALED,
        'log': logzero(TRAIN_TT_SCALED),
        'sqrt': TRAIN_TT_SCALED**0.5,
        'piecewise_1': piecewise_formula(TRAIN_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': piecewise_formula(TRAIN_TT_SCALED, [0, 0.25, None]),
        'boxcox': boxcox(TRAIN_TT_SCALED, ell_tt),
    },
)

sm_tt_catalog = Catalog.from_dict(
    catalog_name='sm_tt_catalog',
    dict_of_expressions={
        'linear': SM_TT_SCALED,
        'log': logzero(SM_TT_SCALED),
        'sqrt': SM_TT_SCALED**0.5,
        'piecewise_1': piecewise_formula(SM_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': piecewise_formula(SM_TT_SCALED, [0, 0.25, None]),
        'boxcox': boxcox(SM_TT_SCALED, ell_tt),
    },
    controlled_by=train_tt_catalog.controlled_by,
)

car_tt_catalog = Catalog.from_dict(
    catalog_name='car_tt_catalog',
    dict_of_expressions={
        'linear': CAR_TT_SCALED,
        'log': logzero(CAR_TT_SCALED),
        'sqrt': CAR_TT_SCALED**0.5,
        'piecewise_1': piecewise_formula(CAR_TT_SCALED, [0, 0.1, None]),
        'piecewise_2': piecewise_formula(CAR_TT_SCALED, [0, 0.25, None]),
        'boxcox': boxcox(CAR_TT_SCALED, ell_tt),
    },
    controlled_by=train_tt_catalog.controlled_by,
)

# %%
# Parameter for Box-Cox transforms.
ell_cost = Beta('lambda_cost', 1, -10, 10, 0)

# %%
# Nonlinear transformations for travel cost.
train_cost_catalog = Catalog.from_dict(
    catalog_name='train_cost_catalog',
    dict_of_expressions={
        'linear': TRAIN_COST_SCALED,
        'log': logzero(TRAIN_COST_SCALED),
        'sqrt': TRAIN_COST_SCALED**0.5,
        'piecewise_1': piecewise_formula(TRAIN_COST_SCALED, [0, 0.1, None]),
        'piecewise_2': piecewise_formula(TRAIN_COST_SCALED, [0, 0.25, None]),
        'boxcox': boxcox(TRAIN_COST_SCALED, ell_cost),
    },
)

sm_cost_catalog = Catalog.from_dict(
    catalog_name='sm_cost_catalog',
    dict_of_expressions={
        'linear': SM_COST_SCALED,
        'log': logzero(SM_COST_SCALED),
        'sqrt': SM_COST_SCALED**0.5,
        'piecewise_1': piecewise_formula(SM_COST_SCALED, [0, 0.1, None]),
        'piecewise_2': piecewise_formula(SM_COST_SCALED, [0, 0.25, None]),
        'boxcox': boxcox(SM_COST_SCALED, ell_cost),
    },
    controlled_by=train_cost_catalog.controlled_by,
)

car_cost_catalog = Catalog.from_dict(
    catalog_name='car_cost_catalog',
    dict_of_expressions={
        'linear': CAR_CO_SCALED,
        'log': logzero(CAR_CO_SCALED),
        'sqrt': CAR_CO_SCALED**0.5,
        'piecewise_1': piecewise_formula(CAR_CO_SCALED, [0, 0.1, None]),
        'piecewise_2': piecewise_formula(CAR_CO_SCALED, [0, 0.25, None]),
        'boxcox': boxcox(CAR_CO_SCALED, ell_cost),
    },
    controlled_by=train_cost_catalog.controlled_by,
)

# %%
# Definition of the utility functions
v_train = (
    asc_train_catalog
    + b_time * train_tt_catalog
    + b_cost * train_cost_catalog
    + train_headway_catalog
)
v_swissmetro = b_time * sm_tt_catalog + b_cost * sm_cost_catalog + sm_headway_catalog

v_car = asc_car_catalog + b_time * car_tt_catalog + b_cost * car_cost_catalog

# %%
# Associate utility functions with the numbering of alternatives
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)

# %%
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b22_multiple_models'

# %%
PARETO_FILE_NAME = 'b22_multiple_models.pareto'
