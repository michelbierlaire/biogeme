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


Michel Bierlaire, EPFL
Sun Apr 27 2025, 15:47:18
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.catalog import Catalog
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
from biogeme.expressions import Beta, Expression
from biogeme.models import boxcox, loglogit
from biogeme.results_processing import compile_estimation_results, pareto_optimal

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, 0, 0)
b_cost = Beta('b_cost', 0, None, 0, 0)

# %%
# Non-linear specifications for the travel time.

# %%
# Parameter of the Box-Cox transform.
lambda_travel_time = Beta('lambda_travel_time', 1, -10, 10, 0)

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
boxcox_train_tt = boxcox(TRAIN_TT_SCALED, lambda_travel_time)

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
boxcox_sm_tt = boxcox(SM_TT_SCALED, lambda_travel_time)

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
boxcox_car_tt = boxcox(CAR_TT_SCALED, lambda_travel_time)

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
v_train = asc_train + b_time * train_tt_catalog + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time * sm_tt_catalog + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * car_tt_catalog + b_cost * CAR_CO_SCALED

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
# Read the data
database = read_data()

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(
    database, log_probability, generate_html=False, generate_yaml=False
)
the_biogeme.model_name = 'b02nonlinear'

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
display('All estimated models')
display(compiled_results)

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
display('Non dominated models')
display(compiled_pareto_results)

# %%
# Glossary.
for short_name, spec in pareto_specs.items():
    print(f'{short_name}\t{spec}')
