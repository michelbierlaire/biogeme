"""
.. _everything_spec_section:

Combination of many specifications
==================================

We combine the following specifications:

- 3 models

    - logit
    - nested logit with two nests: public and private transportation
    - nested logit with two nests existing and future modes

- 3 functional forms for the travel time variables

    - linear specification,
    - Box-Cox transform,
    - power series,

- 2 specifications for the cost coefficients:

    - generic
    - alternative specific

- 2 specifications for the travel time coefficients:

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
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

:author: Michel Bierlaire, EPFL
:date: Sat Jul 15 15:40:33 2023

"""
import numpy as np
from biogeme import models
from biogeme.expressions import Expression, Beta
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
from biogeme.catalog import (
    Catalog,
    segmentation_catalogs,
    generic_alt_specific_catalogs,
)

# %%
# See :ref:`swissmetro_data`.
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
# Definition of the segmentations.
segmentation_ga = database.generate_segmentation(
    variable='GA', mapping={0: 'noGA', 1: 'GA'}
)

segmentation_luggage = database.generate_segmentation(
    variable='LUGGAGE', mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'}
)

segmentation_first = database.generate_segmentation(
    variable='FIRST', mapping={0: '2nd_class', 1: '1st_class'}
)

# %%
# We consider two trip purposes: 'commuters' and anything else. We
# need to define a binary variable first.
database.data['COMMUTERS'] = np.where(database.data['PURPOSE'] == 1, 1, 0)

segmentation_purpose = database.generate_segmentation(
    variable='COMMUTERS', mapping={0: 'non_commuters', 1: 'commuters'}
)

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Parameter of the Box-Cox transform.
ell_travel_time = Beta('lambda_travel_time', 1, -10, 10, 0)

# %%
# Coefficients of the power series.
square_tt_coef = Beta('square_tt_coef', 0, None, None, 0)
cube_tt_coef = Beta('cube_tt_coef', 0, None, None, 0)


# %%
# Function calculation the power series.
def power_series(the_variable: Expression) -> Expression:
    """Generate the expression of a polynomial of degree 3

    :param the_variable: variable of the polynomial
    """
    return (
        the_variable
        + square_tt_coef * the_variable**2
        + cube_tt_coef * the_variable * the_variable**3
    )


# %%
# Train travel time

# %%
# Linear specification.
linear_train_tt = TRAIN_TT_SCALED

# %%
# Box-Cox transform.
boxcox_train_tt = models.boxcox(TRAIN_TT_SCALED, ell_travel_time)

# %%
# Power series.
power_train_tt = power_series(TRAIN_TT_SCALED)

# %%
# Definition of the catalog.
train_tt_catalog = Catalog.from_dict(
    catalog_name='train_tt_catalog',
    dict_of_expressions={
        'linear': linear_train_tt,
        'boxcox': boxcox_train_tt,
        'power': power_train_tt,
    },
)

# %%
# Swissmetro travel time

# %%
# Linear specification.
linear_sm_tt = SM_TT_SCALED

# %%
# Box-Cox transform.
boxcox_sm_tt = models.boxcox(SM_TT_SCALED, ell_travel_time)

# %%
# Power series.
power_sm_tt = power_series(SM_TT_SCALED)

# %%
# Definition of the catalog. Note that the controller is the same as for train.
sm_tt_catalog = Catalog.from_dict(
    catalog_name='sm_tt_catalog',
    dict_of_expressions={
        'linear': linear_sm_tt,
        'boxcox': boxcox_sm_tt,
        'power': power_sm_tt,
    },
    controlled_by=train_tt_catalog.controlled_by,
)

# %%
# Car travel time

# %%
# Linear specification.
linear_car_tt = CAR_TT_SCALED

# %%
# Box-Cox transform.
boxcox_car_tt = models.boxcox(CAR_TT_SCALED, ell_travel_time)

# %%
# Power series.
power_car_tt = power_series(CAR_TT_SCALED)

# %%
# Definition of the catalog. Note that the controller is the same as for train.
car_tt_catalog = Catalog.from_dict(
    catalog_name='car_tt_catalog',
    dict_of_expressions={
        'linear': linear_car_tt,
        'boxcox': boxcox_car_tt,
        'power': power_car_tt,
    },
    controlled_by=train_tt_catalog.controlled_by,
)


# %%
# Catalogs for the alternative specific constants.
ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
    potential_segmentations=(
        segmentation_ga,
        segmentation_luggage,
    ),
    maximum_number=2,
)


# %%
# Catalog for the travel time coefficient.
(B_TIME_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_TIME',
    beta_parameters=[B_TIME],
    alternatives=['TRAIN', 'SM', 'CAR'],
    potential_segmentations=(
        segmentation_first,
        segmentation_purpose,
    ),
    maximum_number=1,
)

# %%
# Catalog for the travel cost coefficient.
(B_COST_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_COST', beta_parameters=[B_COST], alternatives=['TRAIN', 'SM', 'CAR']
)

# %%
# Definition of the utility functions.
V1 = (
    ASC_TRAIN_catalog
    + B_TIME_catalog_dict['TRAIN'] * train_tt_catalog
    + B_COST_catalog_dict['TRAIN'] * TRAIN_COST_SCALED
)
V2 = (
    B_TIME_catalog_dict['SM'] * sm_tt_catalog
    + B_COST_catalog_dict['SM'] * SM_COST_SCALED
)
V3 = (
    ASC_CAR_catalog
    + B_TIME_catalog_dict['CAR'] * car_tt_catalog
    + B_COST_catalog_dict['CAR'] * CAR_CO_SCALED
)

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the logit model. This is the contribution of each
# observation to the log likelihood function.
logprob_logit = models.loglogit(V, av, CHOICE)

# %%
# Nested logit model: nest with existing alternatives.
mu_existing = Beta('mu_existing', 1, 1, 10, 0)
existing = OneNestForNestedLogit(
    nest_param=mu_existing, list_of_alternatives=[1, 3], name='Existing'
)

nests_existing = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(existing,))
logprob_nested_existing = models.lognested(V, av, nests_existing, CHOICE)

# %%
# Nested logit model: nest with public transportation alternatives.
mu_public = Beta('mu_public', 1, 1, 10, 0)
public = OneNestForNestedLogit(
    nest_param=mu_public, list_of_alternatives=[1, 2], name='Public'
)

nests_public = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(public,))
logprob_nested_public = models.lognested(V, av, nests_public, CHOICE)

# %%
# Catalo for models.
model_catalog = Catalog.from_dict(
    catalog_name='model_catalog',
    dict_of_expressions={
        'logit': logprob_logit,
        'nested existing': logprob_nested_existing,
        'nested public': logprob_nested_public,
    },
)
