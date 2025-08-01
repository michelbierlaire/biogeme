"""

Investigation of several choice models
======================================

Investigate several choice models:

    - logit
    - nested logit with two nests: public and private transportation
    - nested logit with two nests existing and future modes

for a total of 3 specifications.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

Michel Bierlaire, EPFL
Sun Apr 27 2025, 15:46:15
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
from biogeme.expressions import Beta
from biogeme.models import loglogit, lognested
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
from biogeme.results_processing import compile_estimation_results, pareto_optimal

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Parameters to be estimated
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Definition of the utility functions
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the logit model. This is the contribution of each
# observation to the log likelihood function.
log_probability_logit = loglogit(v, av, CHOICE)

# %%
# Nested logit model: nest with existing alternatives.
mu_existing = Beta('mu_existing', 1, 1, 10, 0)
existing = OneNestForNestedLogit(
    nest_param=mu_existing, list_of_alternatives=[1, 3], name='Existing'
)

nests_existing = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(existing,))
log_probability_nested_existing = lognested(v, av, nests_existing, CHOICE)

# %%
# Nested logit model: nest with public transportation alternatives.
mu_public = Beta('mu_public', 1, 1, 10, 0)
public = OneNestForNestedLogit(
    nest_param=mu_public, list_of_alternatives=[1, 2], name='Public'
)

nests_public = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(public,))
log_probability_nested_public = lognested(v, av, nests_public, CHOICE)

# %%
# Catalog.
model_catalog = Catalog.from_dict(
    catalog_name='model_catalog',
    dict_of_expressions={
        'logit': log_probability_logit,
        'nested existing': log_probability_nested_existing,
        'nested public': log_probability_nested_public,
    },
)

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, model_catalog, generate_html=False, generate_yaml=False)
the_biogeme.model_name = 'b01model'

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
