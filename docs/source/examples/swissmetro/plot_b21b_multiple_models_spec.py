""".. _plot_b21b_multiple_models_spec:

21b. Specification of a catalog of models
=========================================

Specification of the catalogs used by the assisted specification
algorithm. Note that this script does not perform any estimation. It
is imported by other scripts: :ref:`plot_b21a_multiple_models`, :ref:`plot_b21c_process_pareto`.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 12:09:34
"""

from biogeme.biogeme import BIOGEME
from biogeme.catalog import Catalog, segmentation_catalogs
from biogeme.expressions import Beta, logzero
from biogeme.models import boxcox, loglogit

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    INCOME,
    MALE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Segmentations.
gender_segmentation = database.generate_segmentation(
    variable=MALE,
    mapping={
        0: 'female',
        1: 'male',
    },
)

# %%
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

# %%
ga_segmentation = database.generate_segmentation(
    variable=GA, mapping={1: 'GA', 0: 'noGA'}
)

# %%
asc_segmentations = (
    gender_segmentation,
    ga_segmentation,
)

# %%
asc_car_catalog, asc_train_catalog = segmentation_catalogs(
    generic_name='asc',
    beta_parameters=[asc_car, asc_train],
    potential_segmentations=asc_segmentations,
    maximum_number=2,
)

# %%
cost_segmentations = (
    ga_segmentation,
    income_segmentation,
)

# %%
# Note that the function returns a list. In this case, it contains
# only one element. This is the reason of the presence of the comma
# after B_COST_catalog
(b_cost_catalog,) = segmentation_catalogs(
    generic_name='b_cost',
    beta_parameters=[b_cost],
    potential_segmentations=cost_segmentations,
    maximum_number=1,
)

# %%
# Parameter for Box-Cox transforms
ell_time = Beta('lambda_time', 1, -10, 10, 0)

# %%
# Potential non-linear specifications of travel time.
train_tt_catalog = Catalog.from_dict(
    catalog_name='train_tt',
    dict_of_expressions={
        'linear': TRAIN_TT_SCALED,
        'log': logzero(TRAIN_TT_SCALED),
        'boxcox': boxcox(TRAIN_TT_SCALED, ell_time),
    },
)

sm_tt_catalog = Catalog.from_dict(
    catalog_name='sm_tt',
    dict_of_expressions={
        'linear': SM_TT_SCALED,
        'log': logzero(SM_TT_SCALED),
        'boxcox': boxcox(SM_TT_SCALED, ell_time),
    },
    controlled_by=train_tt_catalog.controlled_by,
)

car_tt_catalog = Catalog.from_dict(
    catalog_name='car_tt',
    dict_of_expressions={
        'linear': CAR_TT_SCALED,
        'log': logzero(CAR_TT_SCALED),
        'boxcox': boxcox(CAR_TT_SCALED, ell_time),
    },
    controlled_by=train_tt_catalog.controlled_by,
)

# %%
# Definition of the utility functions with linear cost.
v_train = (
    asc_train_catalog + b_time * train_tt_catalog + b_cost_catalog * TRAIN_COST_SCALED
)
v_swissmetro = b_time * sm_tt_catalog + b_cost_catalog * SM_COST_SCALED
v_car = asc_car_catalog + b_time * car_tt_catalog + b_cost_catalog * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)


# %%
# Create the biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b21_multiple_models'

# %%
# Name of the Pareto file.
PARETO_FILE_NAME = 'b21_multiple_models.pareto'
