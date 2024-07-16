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


:author: Michel Bierlaire, EPFL
:date: Thu Jul 13 16:18:10 2023

"""
import biogeme.biogeme as bio
import biogeme.biogeme_logging as blog
from biogeme import models
from biogeme.expressions import Beta
from biogeme.catalog import generic_alt_specific_catalogs
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
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Catalog for travel time coefficient.
(B_TIME_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_TIME', beta_parameters=[B_TIME], alternatives=('TRAIN', 'SM', 'CAR')
)

# %%
# Catalog for travel cost coefficient.
(B_COST_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_COST', beta_parameters=[B_COST], alternatives=('TRAIN', 'SM', 'CAR')
)

# %%
# Definition of the utility functions.
V1 = (
    ASC_TRAIN
    + B_TIME_catalog_dict['TRAIN'] * TRAIN_TT_SCALED
    + B_COST_catalog_dict['TRAIN'] * TRAIN_COST_SCALED
)
V2 = (
    B_TIME_catalog_dict['SM'] * SM_TT_SCALED
    + B_COST_catalog_dict['SM'] * SM_COST_SCALED
)
V3 = (
    ASC_CAR
    + B_TIME_catalog_dict['CAR'] * CAR_TT_SCALED
    + B_COST_catalog_dict['CAR'] * CAR_CO_SCALED
)

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
the_biogeme.modelName = 'b01alt_spec'
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
