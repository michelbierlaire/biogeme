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

:author: Michel Bierlaire, EPFL
:date: Fri Jul 14 09:47:21 2023

"""
import biogeme.biogeme as bio
import biogeme.biogeme_logging as blog
from biogeme import models
from biogeme.expressions import Beta
from biogeme.catalog import Catalog
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
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
# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives
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
# Catalog.
model_catalog = Catalog.from_dict(
    catalog_name='model_catalog',
    dict_of_expressions={
        'logit': logprob_logit,
        'nested existing': logprob_nested_existing,
        'nested public': logprob_nested_public,
    },
)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b01model'
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
