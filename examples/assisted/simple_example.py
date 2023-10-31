"""File simple_example.py

:author: Michel Bierlaire, EPFL
:date: Sun Aug  6 18:13:18 2023

Example of a catalog

"""
import sys
import numpy as np
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, Expression
from biogeme.models import boxcox
from biogeme.catalog import Catalog, generic_alt_specific_catalogs, segmentation_catalogs
from results_analysis import report
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

def print_all_configurations(expression: Expression) -> None:
    """Prints all configurations that an expression can take
    """
    expression.set_central_controller()
    total = expression.central_controller.number_of_configurations()
    print(f'Total: {total} configurations')
    for config_id in expression.central_controller.all_configurations_ids:
        print(config_id)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)


# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob_logit = models.loglogit(V, av, CHOICE)

MU = Beta('MU', 1, 1, 10, 0)
existing = MU, [1, 3]
future = 1.0, [2]
nests = existing, future
logprob_nested = models.lognested(V, av, nests, CHOICE)

model_catalog = Catalog.from_dict(
    catalog_name='model_catalog',
    dict_of_expressions={
        'logit': logprob_logit,
        'nested': logprob_nested,
    },
)

print('*** Current status of the catalog ***')
print(model_catalog)
print('*** Use the controller to select a different configuration ***')
model_catalog.controlled_by.set_name('nested')
print('*** Current status of the catalog ***')
print(model_catalog)

print('*** Iterator ***')
for specification in model_catalog:
    print(specification)

print_all_configurations(model_catalog)
    
print('*** Nonlinear specifications *** ')
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

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, 0, 0)
V_TRAIN = ASC_TRAIN + B_TIME * train_tt_catalog

print_all_configurations(V_TRAIN)

print('** Unsynchronized catalogs **')
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
    
dummy_expression = train_tt_catalog + car_tt_catalog

print_all_configurations(dummy_expression)

print('** Synchronized catalogs **')
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
    controlled_by=train_tt_catalog.controlled_by
)
    
dummy_expression = train_tt_catalog + car_tt_catalog

print_all_configurations(dummy_expression)


print('*** Alternative specific ***')

(B_TIME_catalog_dict, B_COST_catalog_dict) = generic_alt_specific_catalogs(
    generic_name='coefficients',
    beta_parameters=[B_TIME, B_COST],
    alternatives=('TRAIN', 'CAR')
)

V_TRAIN = (
    B_TIME_catalog_dict['TRAIN'] * TRAIN_TT +
    B_COST_catalog_dict['TRAIN'] * TRAIN_COST
)
V_CAR = (
    B_TIME_catalog_dict['CAR'] * CAR_TT +
    B_COST_catalog_dict['CAR'] * CAR_COST
)

dummy_expression = V_TRAIN + V_CAR

print_all_configurations(dummy_expression)

print('*** Alternative specific - not synchronized ***')
    
(B_TIME_catalog_dict, ) = generic_alt_specific_catalogs(
    generic_name='time_coefficient',
    beta_parameters=[B_TIME],
    alternatives=('TRAIN', 'CAR')
)

(B_COST_catalog_dict, ) = generic_alt_specific_catalogs(
    generic_name='cost_coefficient',
    beta_parameters=[B_COST],
    alternatives=('TRAIN', 'CAR')
)

V_TRAIN = (
    B_TIME_catalog_dict['TRAIN'] * TRAIN_TT +
    B_COST_catalog_dict['TRAIN'] * TRAIN_COST
)
V_CAR = (
    B_TIME_catalog_dict['CAR'] * CAR_TT +
    B_COST_catalog_dict['CAR'] * CAR_COST
)

dummy_expression = V_TRAIN + V_CAR

print_all_configurations(dummy_expression)

print('*** Segmentation ***')

# We consider two trip purposes: 'commuters' and anything else. We
# need to define a binary variable first
database.data['COMMUTERS'] = np.where(database.data['PURPOSE'] == 1, 1, 0)
segmentation_purpose = database.generate_segmentation(
    variable='COMMUTERS',
    mapping={
        0: 'non_commuters',
        1: 'commuters'
    },
    reference='non_commuters'
    
)
segmentation_luggage = database.generate_segmentation(
    variable='LUGGAGE',
    mapping={
        0: 'no_lugg',
        1: 'one_lugg',
        3: 'several_lugg'
    },
    reference='no_lugg'
)


ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=2,
)



dummy_expression = ASC_TRAIN_catalog + ASC_CAR_catalog

print_all_configurations(dummy_expression)

ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
    potential_segmentations=(
        segmentation_purpose,
        segmentation_luggage,
    ),
    maximum_number=1,
)



dummy_expression = ASC_TRAIN_catalog + ASC_CAR_catalog

print_all_configurations(dummy_expression)

print('** Segmentation and alternative specific **')

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

print_all_configurations(B_TIME_catalog_dict['TRAIN'])

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

print_all_configurations(B_TIME_catalog_dict['TRAIN'])
