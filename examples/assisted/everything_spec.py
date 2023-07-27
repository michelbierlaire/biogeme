"""File everything_spec.py

:author: Michel Bierlaire, EPFL
:date: Sat Jul 15 15:40:33 2023

We investigate various specifications:
- 3 models
    - logit
    - nested logit with two nests: public and private transportation
    - nested logit with two nests existing and future modes
- 3 functional form for the travel time variables
    - linear specification,
    - Box-Cox transform,
    - power series,
- 2 specification for the cost coefficients:
    - generic
    - alternative specific
- 2 specification for the travel time coefficients:
    - generic
    - alternative specific
- 4 segmentations for the constants:
    - not segmented
    - segmented by GA (yearly subscription to public transport)
    - segmented by luggage
    - segmented both by GA and luggage
-  3 segmentations for the time coefficients:
    - not segmented
    - segmented with first class
    - segmented with trip purpose

This leads to a total of 432 specifications.
"""
import numpy as np
from biogeme import models
from biogeme.expressions import Beta
from biogeme.catalog import (
    Catalog,
    segmentation_catalogs,
    generic_alt_specific_catalogs,
)

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

segmentation_ga = database.generate_segmentation(
    variable='GA', mapping={0: 'noGA', 1: 'GA'}
)

segmentation_luggage = database.generate_segmentation(
    variable='LUGGAGE', mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'}
)

segmentation_first = database.generate_segmentation(
    variable='FIRST', mapping={0: '2nd_class', 1: '1st_class'}
)

# We consider two trip purposes: 'commuters' and anything else. We
# need to define a binary variable first

database.data['COMMUTERS'] = np.where(database.data['PURPOSE'] == 1, 1, 0)

segmentation_purpose = database.generate_segmentation(
    variable='COMMUTERS', mapping={0: 'non_commuters', 1: 'commuters'}
)


# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Non linear specifications for the travel time

# Parameter of the Box-Cox transform
ell_travel_time = Beta('lambda_travel_time', 1, None, 10, 0)

# Coefficients of the power series
square_tt_coef = Beta('square_tt_coef', 0, None, None, 0)
cube_tt_coef = Beta('cube_tt_coef', 0, None, None, 0)


def power_series(the_variable):
    """Generate the expression of a polynomial of degree 3

    :param the_variable: variable of the polynomial
    :type the_variable: biogeme.expressions.Expression
    """
    return (
        the_variable
        + square_tt_coef * the_variable**2
        + cube_tt_coef * the_variable * the_variable**3
    )


linear_train_tt = TRAIN_TT_SCALED
boxcox_train_tt = models.boxcox(TRAIN_TT_SCALED, ell_travel_time)
power_train_tt = power_series(TRAIN_TT_SCALED)
train_tt_catalog = Catalog.from_dict(
    catalog_name='train_tt_catalog',
    dict_of_expressions={
        'linear': linear_train_tt,
        'boxcox': boxcox_train_tt,
        'power': power_train_tt,
    },
)

linear_sm_tt = SM_TT_SCALED
boxcox_sm_tt = models.boxcox(SM_TT_SCALED, ell_travel_time)
power_sm_tt = power_series(SM_TT_SCALED)
sm_tt_catalog = Catalog.from_dict(
    catalog_name='sm_tt_catalog',
    dict_of_expressions={
        'linear': linear_sm_tt,
        'boxcox': boxcox_sm_tt,
        'power': power_sm_tt,
    },
    controlled_by=train_tt_catalog.controlled_by,
)

linear_car_tt = CAR_TT_SCALED
boxcox_car_tt = models.boxcox(CAR_TT_SCALED, ell_travel_time)
power_car_tt = power_series(CAR_TT_SCALED)

car_tt_catalog = Catalog.from_dict(
    catalog_name='car_tt_catalog',
    dict_of_expressions={
        'linear': linear_car_tt,
        'boxcox': boxcox_car_tt,
        'power': power_car_tt,
    },
    controlled_by=train_tt_catalog.controlled_by,
)


ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
    potential_segmentations=(
        segmentation_ga,
        segmentation_luggage,
    ),
    maximum_number=2,
)


(B_TIME_catalog,) = generic_alt_specific_catalogs(
    generic_name='B_TIME',
    beta_parameters=[B_TIME],
    alternatives=['TRAIN', 'SM', 'CAR'],
    potential_segmentations=(
        segmentation_first,
        segmentation_purpose,
    ),
    maximum_number=1,
)

(B_COST_catalog,) = generic_alt_specific_catalogs(
    generic_name='B_COST', beta_parameters=[B_COST], alternatives=['TRAIN', 'SM', 'CAR']
)

# Definition of the utility functions
V1 = (
    ASC_TRAIN_catalog
    + B_TIME_catalog['TRAIN'] * train_tt_catalog
    + B_COST_catalog['TRAIN'] * TRAIN_COST_SCALED
)
V2 = B_TIME_catalog['SM'] * sm_tt_catalog + B_COST_catalog['SM'] * SM_COST_SCALED
V3 = (
    ASC_CAR_catalog
    + B_TIME_catalog['CAR'] * car_tt_catalog
    + B_COST_catalog['CAR'] * CAR_CO_SCALED
)

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob_logit = models.loglogit(V, av, CHOICE)

MU_existing = Beta('MU_existing', 1, 1, 10, 0)
existing = MU_existing, [1, 3]
future = 1.0, [2]
nests_existing = existing, future
logprob_nested_existing = models.lognested(V, av, nests_existing, CHOICE)

MU_public = Beta('MU_public', 1, 1, 10, 0)
public = MU_public, [1, 2]
private = 1.0, [3]
nests_public = public, private
logprob_nested_public = models.lognested(V, av, nests_public, CHOICE)

model_catalog = Catalog.from_dict(
    catalog_name='model_catalog',
    dict_of_expressions={
        'logit': logprob_logit,
        'nested existing': logprob_nested_existing,
        'nested public': logprob_nested_public,
    },
)
