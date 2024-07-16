"""

Example of a catalog
====================

Illustration of the concept of catalog. See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_

:author: Michel Bierlaire, EPFL
:date: Sun Aug  6 18:13:18 2023

"""
import numpy as np
from biogeme import models
from biogeme.expressions import Beta, Variable, Expression
from biogeme.models import boxcox
from biogeme.catalog import (
    Catalog,
    generic_alt_specific_catalogs,
    segmentation_catalogs,
)
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit

# %%
# See :ref:`swissmetro_data`
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
)


# %%
# Function printing all configurations of an expression.
def print_all_configurations(expression: Expression) -> None:
    """Prints all configurations that an expression can take"""
    expression.set_central_controller()
    total = expression.central_controller.number_of_configurations()
    print(f'Total: {total} configurations')
    for config_id in expression.central_controller.all_configurations_ids:
        print(config_id)


# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob_logit = models.loglogit(V, av, CHOICE)

# %%
# Nest definition.

mu_existing = Beta('mu_existing', 1, 1, 10, 0)
existing = OneNestForNestedLogit(nest_param=mu_existing, list_of_alternatives=[1, 3])
nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(existing,))

# %%
# Contribution to the log-likelihood.
logprob_nested = models.lognested(V, av, nests, CHOICE)

# %%
# Definition of the catalog containnig two models specifications:
# logit and nested logit.
model_catalog = Catalog.from_dict(
    catalog_name='model_catalog',
    dict_of_expressions={
        'logit': logprob_logit,
        'nested': logprob_nested,
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

# %% Non linear specifications.
TRAIN_TT = Variable('TRAIN_TT')
TRAIN_COST = Variable('TRAIN_COST')
ell_travel_time = Beta('lambda_travel_time', 1, -10, 10, 0)
linear_train_tt = TRAIN_TT
boxcox_train_tt = boxcox(TRAIN_TT, ell_travel_time)
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
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, 0, 0)
V_TRAIN = ASC_TRAIN + B_TIME * train_tt_catalog

# %%
print_all_configurations(V_TRAIN)

# %%
# Unsynchronized catalogs
CAR_TT = Variable('CAR_TT')
CAR_COST = Variable('CAR_COST')
linear_car_tt = CAR_TT
boxcox_car_tt = boxcox(CAR_TT, ell_travel_time)
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
CAR_TT = Variable('CAR_TT')
CAR_COST = Variable('CAR_COST')
linear_car_tt = CAR_TT
boxcox_car_tt = boxcox(CAR_TT, ell_travel_time)
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

(B_TIME_catalog_dict, B_COST_catalog_dict) = generic_alt_specific_catalogs(
    generic_name='coefficients',
    beta_parameters=[B_TIME, B_COST],
    alternatives=('TRAIN', 'CAR'),
)

# %%
# Create utility functions involving those catalogs.
V_TRAIN = (
    B_TIME_catalog_dict['TRAIN'] * TRAIN_TT + B_COST_catalog_dict['TRAIN'] * TRAIN_COST
)
V_CAR = B_TIME_catalog_dict['CAR'] * CAR_TT + B_COST_catalog_dict['CAR'] * CAR_COST

# %%
# Create a dummy expression involving the utility functions.
dummy_expression = V_TRAIN + V_CAR

# %%
print_all_configurations(dummy_expression)

# %%
# Alternative specific - not synchronized.

(B_TIME_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='time_coefficient',
    beta_parameters=[B_TIME],
    alternatives=('TRAIN', 'CAR'),
)

(B_COST_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='cost_coefficient',
    beta_parameters=[B_COST],
    alternatives=('TRAIN', 'CAR'),
)

# %%
# Create utility functions involving those catalogs.
V_TRAIN = (
    B_TIME_catalog_dict['TRAIN'] * TRAIN_TT + B_COST_catalog_dict['TRAIN'] * TRAIN_COST
)
V_CAR = B_TIME_catalog_dict['CAR'] * CAR_TT + B_COST_catalog_dict['CAR'] * CAR_COST

# %%
# Create a dummy expression involving the utility functions.
dummy_expression = V_TRAIN + V_CAR

# %%
print_all_configurations(dummy_expression)

# %%
# Segmentation

# %%
# We consider two trip purposes: `commuters` and anything else. We
# need to define a binary variable first.
database.data['COMMUTERS'] = np.where(database.data['PURPOSE'] == 1, 1, 0)

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
ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=2,
)

# %%
# Create a dummy expression.
dummy_expression = ASC_TRAIN_catalog + ASC_CAR_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Catalog of segmented alternative specific constants, allows a maximum
# of one segmentation.
ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=1,
)

# %%
# Create a dummy expression.
dummy_expression = ASC_TRAIN_catalog + ASC_CAR_catalog

# %%
print_all_configurations(dummy_expression)

# %%
# Segmentation and alternative specific
# Maximum one segmentation.
(B_TIME_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_TIME',
    beta_parameters=[B_TIME],
    alternatives=['TRAIN', 'CAR'],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=1,
)

# %%
print_all_configurations(B_TIME_catalog_dict['TRAIN'])

# %%
# Maximum two segmentations.
(B_TIME_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_TIME',
    beta_parameters=[B_TIME],
    alternatives=['TRAIN', 'CAR'],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=2,
)

# %%
print_all_configurations(B_TIME_catalog_dict['TRAIN'])
