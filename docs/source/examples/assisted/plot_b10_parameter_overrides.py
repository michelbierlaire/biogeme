""".. _plot_b10_parameter_overrides:

10. Controlling a generated parameter for a missing segmentation category
==========================================================================

This example reproduces a common assisted-specification problem.  A variable
``has_pt_subscr`` is coded as 1 or 2, but one observation uses ``-99`` for a
missing value.  ``Database.generate_segmentation`` automatically creates a
category for that value, and the segmentation catalogs consequently contain
parameters for the ``minus_99`` category.

The example fixes every generated ``minus_99`` coefficient to zero with
``ParameterOverrides``.  The override is applied to the complete catalog
expression before it is sent to Biogeme, so every catalog alternative is
handled consistently.

The Swissmetro data do not contain a public-transport-subscription variable.
For documentation purposes, this script derives one from ``GA`` and marks one
observation as ``-99``.  The construction is only to reproduce the user's
case; in an application, the column is read from the user's database.

Michel Bierlaire, EPFL
"""

from __future__ import annotations

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.catalog import segmentation_catalogs
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    read_data,
)
from biogeme.expressions import (
    Beta,
    Numeric,
    ParameterOverrides,
    apply_parameter_overrides,
    list_of_all_betas_in_expression,
)
from biogeme.models import loglogit

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example plot_b10_parameter_overrides.py')

# %%
# Load the Swissmetro data and create a didactic subscription variable.  The
# user's original coding is retained: 1 means subscription, 2 means no
# subscription, and -99 is an observed missing value.
database = read_data()
database.dataframe['has_pt_subscr'] = database.dataframe['GA'].map({0: 2, 1: 1})
database.dataframe.loc[database.dataframe.index[0], 'has_pt_subscr'] = -99

segmentation_pt_subscription = database.generate_segmentation(
    variable='has_pt_subscr',
    mapping={2: 'no_pt_subscr', 1: 'pt_subscr', -99: 'minus_99'},
    reference='no_pt_subscr',
)

# %%
# Build a small assisted-specification catalog.  The catalog includes the
# automatically generated ``minus_99`` segment in every segmented
# alternative-specific constant.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

asc_train_catalog, asc_car_catalog = segmentation_catalogs(
    generic_name='asc',
    beta_parameters=[asc_train, asc_car],
    potential_segmentations=(segmentation_pt_subscription,),
    maximum_number=1,
)

v_train = asc_train_catalog + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car_catalog + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

utilities = {1: v_train, 2: v_swissmetro, 3: v_car}
availability = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
log_probability = loglogit(utilities, availability, CHOICE)

# Select the segmented alternative while retaining all catalog branches in the
# expression.  This makes the generated missing-category coefficients active
# in the model estimated below.
asc_train_catalog.controlled_by.set_name('has_pt_subscr')

# %%
# Locate the generated parameters by their actual Beta names.  In a real model
# the same names can be read from the expression or from the generated catalog
# code; they are not guessed from the catalog labels.
missing_parameter_names = sorted(
    {
        beta.name
        for beta in list_of_all_betas_in_expression(log_probability)
        if beta.name.endswith('_minus_99')
    }
)
if not missing_parameter_names:
    raise RuntimeError('The didactic minus_99 segment did not generate any parameters.')

overrides = ParameterOverrides()
for parameter_name in missing_parameter_names:
    overrides.set(parameter_name, Numeric(0))

log_probability = apply_parameter_overrides(log_probability, overrides)

remaining_missing_parameters = {
    beta.name
    for beta in list_of_all_betas_in_expression(log_probability)
    if beta.name.endswith('_minus_99')
}
print(f'Generated minus_99 parameters: {missing_parameter_names}')
print(f'Parameters remaining after overrides: {sorted(remaining_missing_parameters)}')

# %%
# Estimate the selected segmented model.  The missing-category coefficients are
# now fixed at zero instead of being estimated from a single observation.
biogeme = BIOGEME(database, log_probability)
biogeme.model_name = 'b10_parameter_overrides'
results = biogeme.estimate()
print(results.short_summary())
