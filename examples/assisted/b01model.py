"""File b01model.py

:author: Michel Bierlaire, EPFL
:date: Fri Jul 14 09:47:21 2023

Investigate several choice models:
- logit
- nested logit with two nests: public and private transportation
- nested logit with two nests existing and future modes
for a total of 3 specifications.
"""
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from biogeme.catalog import Catalog
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
# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'b05model'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# Estimate the parameters
dict_of_results = the_biogeme.estimate_catalog()

report(dict_of_results)
