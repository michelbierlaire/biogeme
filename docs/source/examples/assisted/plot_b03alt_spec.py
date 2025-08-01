"""

Catalog for alternative specific coefficients
=============================================

Investigate alternative specific parameters:

    - two specifications for the travel time coefficient: generic, and
      alternative specific,
    - two specifications for the travel cost coefficient: generic, and
      alternative specific,

for a total of 4 specifications.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.


Michel Bierlaire, EPFL
Sun Apr 27 2025, 15:49:05
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.catalog import generic_alt_specific_catalogs
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
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import compile_estimation_results, pareto_optimal

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Catalog for travel time coefficient.
(b_time_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='b_time',
    beta_parameters=[b_time],
    alternatives=('train', 'swissmetro', 'car'),
)

# %%
# Catalog for travel cost coefficient.
(b_cost_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='b_cost',
    beta_parameters=[b_cost],
    alternatives=('train', 'swissmetro', 'car'),
)

# %%
# Definition of the utility functions.
v_train = (
    asc_train
    + b_time_catalog_dict['train'] * TRAIN_TT_SCALED
    + b_cost_catalog_dict['train'] * TRAIN_COST_SCALED
)
v_swissmetro = (
    b_time_catalog_dict['swissmetro'] * SM_TT_SCALED
    + b_cost_catalog_dict['swissmetro'] * SM_COST_SCALED
)
v_car = (
    asc_car
    + b_time_catalog_dict['car'] * CAR_TT_SCALED
    + b_cost_catalog_dict['car'] * CAR_CO_SCALED
)

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
the_biogeme.model_name = 'b01alt_spec'

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
