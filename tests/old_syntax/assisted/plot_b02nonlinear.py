"""

Catalog of nonlinear specifications
===================================

Investigate of nonlinear specifications for the travel time variables:

    - linear specification,
    - Box-Cox transform,
    - power series,

for a total of 3 specifications.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.


:author: Michel Bierlaire, EPFL
:date: Thu Jul 13 21:31:54 2023

"""
import biogeme.biogeme as bio
import biogeme.biogeme_logging as blog
from biogeme import models
from biogeme.expressions import Expression, Beta
from biogeme.models import boxcox
from biogeme.catalog import Catalog
from biogeme.results import compile_estimation_results, pareto_optimal

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

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, 0, 0)
B_COST = Beta('B_COST', 0, None, 0, 0)

# %%
# Non linear specifications for the travel time.

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
boxcox_train_tt = boxcox(TRAIN_TT_SCALED, ell_travel_time)

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
boxcox_sm_tt = boxcox(SM_TT_SCALED, ell_travel_time)

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
boxcox_car_tt = boxcox(CAR_TT_SCALED, ell_travel_time)

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
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * train_tt_catalog + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * sm_tt_catalog + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * car_tt_catalog + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b02nonlinear'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# %%
# Estimate the parameters.
dict_of_results = the_biogeme.estimate_catalog()

# %%
# Number of estimated models.
print(f'A total of {len(dict_of_results)} models have been estimated')

# %%
# All estimation results
compiled_results, specs = compile_estimation_results(
    dict_of_results, use_short_names=True
)

# %%
compiled_results

# %%
# Glossary
for short_name, spec in specs.items():
    print(f'{short_name}\t{spec}')

# %%
# Estimation results of the Pareto optimal models.
pareto_results = pareto_optimal(dict_of_results)
compiled_pareto_results, pareto_specs = compile_estimation_results(
    pareto_results, use_short_names=True
)

# %%
compiled_pareto_results

# %%
# Glossary.
for short_name, spec in pareto_specs.items():
    print(f'{short_name}\t{spec}')
