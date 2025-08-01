"""

Example of a catalog
====================

Illustration of the concept of catalog. See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_

Michel Bierlaire, EPFL
Sun Apr 27 2025, 18:39:23
"""

import numpy as np

from biogeme.catalog import (
    Catalog,
    CentralController,
    generic_alt_specific_catalogs,
    segmentation_catalogs,
)
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO,
    CAR_CO_SCALED,
    CAR_TT,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST,
    TRAIN_COST_SCALED,
    TRAIN_TT,
    TRAIN_TT_SCALED,
    read_data,
)
from biogeme.expressions import Beta, Expression
from biogeme.models import boxcox, loglogit, lognested
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit


# %%
# Function printing all configurations of an expression.
def print_all_configurations(expression: Expression) -> None:
    """Prints all configurations that an expression can take"""
    the_central_controller = CentralController(expression=expression)
    total = the_central_controller.number_of_configurations()
    print(f'Total: {total} configurations')
    for config_id in the_central_controller.all_configurations_ids:
        print(config_id)


# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
log_probability_logit = loglogit(v, av, CHOICE)

# %%
# Nest definition.

mu_existing = Beta('mu_existing', 1, 1, 10, 0)
existing = OneNestForNestedLogit(nest_param=mu_existing, list_of_alternatives=[1, 3])
nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(existing,))

# %%
# Contribution to the log-likelihood.
log_probability_nested = lognested(v, av, nests, CHOICE)

# %%
# Definition of the catalog containing two models specifications:
# logit and nested logit.
model_catalog = Catalog.from_dict(
    catalog_name='model_catalog',
    dict_of_expressions={
        'logit': log_probability_logit,
        'nested': log_probability_nested,
    },
)

# %%
# Current status of the catalog.
print(model_catalog)

# %%
# Use the controller to select a different configuration.
model_catalog.controlled_by.set_name('nested')
print(model_catalog)

# %%
# Iterator.
for specification in model_catalog:
    print(specification)

# %%
# All configurations.
print_all_configurations(model_catalog)

# %% Non-linear specifications.
lambda_travel_time = Beta('lambda_travel_time', 1, -10, 10, 0)
linear_train_tt = TRAIN_TT
boxcox_train_tt = boxcox(TRAIN_TT, lambda_travel_time)
squared_train_tt = TRAIN_TT * TRAIN_TT
train_tt_catalog = Catalog.from_dict(
    catalog_name='train_tt_catalog',
    dict_of_expressions={
        'linear': linear_train_tt,
        'boxcox': boxcox_train_tt,
        'squared': squared_train_tt,
    },
)

# %%
# Define a utility function involving the catalog.
asc_train = Beta('ASC_TRAIN', 0, None, None, 0)
b_time = Beta('B_TIME', 0, None, 0, 0)
v_train_catalog = asc_train + b_time * train_tt_catalog

# %%
print_all_configurations(v_train_catalog)

# %%
# Unsynchronized catalogs
linear_car_tt = CAR_TT
boxcox_car_tt = boxcox(CAR_TT, lambda_travel_time)
squared_car_tt = CAR_TT * CAR_TT
car_tt_catalog = Catalog.from_dict(
    catalog_name='car_tt_catalog',
    dict_of_expressions={
        'linear': linear_car_tt,
        'boxcox': boxcox_car_tt,
        'squared': squared_car_tt,
    },
)

# %%
# Create a dummy expression with the two catalogs.
dummy_expression = train_tt_catalog + car_tt_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Synchronized catalogs.
linear_car_tt = CAR_TT
boxcox_car_tt = boxcox(CAR_TT, lambda_travel_time)
squared_car_tt = CAR_TT * CAR_TT
car_tt_catalog = Catalog.from_dict(
    catalog_name='car_tt_catalog',
    dict_of_expressions={
        'linear': linear_car_tt,
        'boxcox': boxcox_car_tt,
        'squared': squared_car_tt,
    },
    controlled_by=train_tt_catalog.controlled_by,
)

# %%
# Create a dummy expression with the two catalogs.
dummy_expression = train_tt_catalog + car_tt_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Alternative specific specification.

(b_time_catalog_dict, b_cost_catalog_dict) = generic_alt_specific_catalogs(
    generic_name='coefficients',
    beta_parameters=[b_time, b_cost],
    alternatives=('train', 'car'),
)

# %%
# Create utility functions involving those catalogs.
v_train_catalog = (
    b_time_catalog_dict['train'] * TRAIN_TT + b_cost_catalog_dict['train'] * TRAIN_COST
)
v_car_catalog = (
    b_time_catalog_dict['car'] * CAR_TT + b_cost_catalog_dict['car'] * CAR_CO
)

# %%
# Create a dummy expression involving the utility functions.
dummy_expression = v_train_catalog + v_car_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Alternative specific - not synchronized.
(b_time_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='time_coefficient',
    beta_parameters=[b_time],
    alternatives=('train', 'car'),
)

(b_cost_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='cost_coefficient',
    beta_parameters=[b_cost],
    alternatives=('train', 'car'),
)

# %%
# Create utility functions involving those catalogs.
v_train_catalog = (
    b_time_catalog_dict['train'] * TRAIN_TT + b_cost_catalog_dict['train'] * TRAIN_COST
)
v_car_catalog = (
    b_time_catalog_dict['car'] * CAR_TT + b_cost_catalog_dict['car'] * CAR_CO
)

# %%
# Create a dummy expression involving the utility functions.
dummy_expression = v_train_catalog + v_car_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Read the data
database = read_data()

# %%
# Segmentation

# %%
# We consider two trip purposes: `commuters` and anything else. We
# need to define a binary variable first.
database.dataframe['COMMUTERS'] = np.where(database.dataframe['PURPOSE'] == 1, 1, 0)

# %%
# Segmentation on trip purpose.
segmentation_purpose = database.generate_segmentation(
    variable='COMMUTERS',
    mapping={0: 'non_commuters', 1: 'commuters'},
    reference='non_commuters',
)

# %%
# Segmentation on luggage.
segmentation_luggage = database.generate_segmentation(
    variable='LUGGAGE',
    mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'},
    reference='no_lugg',
)

# %%
# Catalog of segmented alternative specific constants, allows a maximum
# of two segmentations.
asc_train_catalog, asc_car_catalog = segmentation_catalogs(
    generic_name='asc',
    beta_parameters=[asc_train, asc_car],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=2,
)

# %%
# Create a dummy expression.
dummy_expression = asc_train_catalog + asc_car_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Catalog of segmented alternative specific constants, allows a maximum
# of one segmentation.
asc_train_catalog, asc_car_catalog = segmentation_catalogs(
    generic_name='asc',
    beta_parameters=[asc_train, asc_car],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=1,
)

# %%
# Create a dummy expression.
dummy_expression = asc_train_catalog + asc_car_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Segmentation and alternative specific
# Maximum one segmentation.
(b_time_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='b_time',
    beta_parameters=[b_time],
    alternatives=('train', 'car'),
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=1,
)

# %%
print_all_configurations(b_time_catalog_dict['train'])

# %%
# Maximum two segmentations.
(b_time_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='b_time',
    beta_parameters=[b_time],
    alternatives=('train', 'car'),
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=2,
)

# %%
print_all_configurations(b_time_catalog_dict['train'])
