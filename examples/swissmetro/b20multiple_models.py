"""File b20multiple_models.py

:author: Michel Bierlaire, EPFL
:date: Mon Apr 10 12:19:46 2023

 Example of the estimation of several specifications of the model

"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, log
from biogeme.results import compile_estimation_results, pareto_optimal
from biogeme.catalog import Catalog, segmentation_catalogs
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
    MALE,
)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_CAR_MALE = Beta('ASC_CAR_MALE', 0, None, None, 0)
ASC_CAR_FEMALE = Beta('ASC_CAR_FEMALE', 0, None, None, 0)
ASC_TRAIN_MALE = Beta('ASC_TRAIN_MALE', 0, None, None, 0)
ASC_TRAIN_FEMALE = Beta('ASC_TRAIN_FEMALE', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

segmentation_gender = database.generate_segmentation(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)


# We define catalogs with two different specifications for the
# ASC_CAR: non segmented, and segmented
ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
    potential_segmentations=(segmentation_gender,),
    maximum_number=1,
)

# We now define a catalog  with the log travel time as well as the travel time
# First for train
train_tt_catalog = Catalog.from_dict(
    catalog_name='train_tt_catalog',
    dict_of_expressions={
        'linear': TRAIN_TT_SCALED,
        'log': log(TRAIN_TT_SCALED),
    },
)

# Then for SM. But we require that the specification is the same as
# train by defining the same controller.
sm_tt_catalog = Catalog.from_dict(
    catalog_name='sm_tt_catalog',
    dict_of_expressions={
        'linear': SM_TT_SCALED,
        'log': log(SM_TT_SCALED),
    },
    controlled_by=train_tt_catalog.controlled_by,
)


# Definition of the utility functions with linear cost
V1 = ASC_TRAIN_catalog + B_TIME * train_tt_catalog + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * sm_tt_catalog + B_COST * SM_COST_SCALED
V3 = ASC_CAR_catalog + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b20multiple_models'
dict_of_results = the_biogeme.estimate_catalog()

print(f'A total of {len(dict_of_results)} models have been estimated:')
for config, res in dict_of_results.items():
    print(f'{config}: LL={res.data.logLike:.2f} K={res.data.nparam}')

summary, description = compile_estimation_results(dict_of_results, use_short_names=True)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')

non_dominated_models = pareto_optimal(dict_of_results)
print(f'Out of them, {len(non_dominated_models)} are non dominated.')
for config, res in non_dominated_models.items():
    print(f'{config}')

summary, description = compile_estimation_results(non_dominated_models)
print(summary)
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')
