"""

Estimation of several models
============================

Example of the estimation of several specifications of the model.

Michel Bierlaire, EPFL
Thu Jun 26 2025, 16:04:27
"""

from biogeme.biogeme import BIOGEME
from biogeme.catalog import (
    Catalog,
    segmentation_catalogs,
)
from biogeme.expressions import Beta, log
from biogeme.models import loglogit
from biogeme.results_processing import (
    compare_parameters,
    compile_estimation_results,
    pareto_optimal,
)

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    MALE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

# %%
# Parameters to be estimated
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
segmentation_gender = database.generate_segmentation(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

# %%
# We define catalogs with two different specifications for the
# ASC_CAR: non segmented, and segmented.
asc_train_catalog, asc_car_catalog = segmentation_catalogs(
    generic_name='asc',
    beta_parameters=[asc_train, asc_car],
    potential_segmentations=(segmentation_gender,),
    maximum_number=1,
)

# %%
# We now define a catalog  with the log travel time as well as the travel time.

# %%
# First for train
train_tt_catalog = Catalog.from_dict(
    catalog_name='train_tt_catalog',
    dict_of_expressions={
        'linear': TRAIN_TT_SCALED,
        'log': log(TRAIN_TT_SCALED),
    },
)

# %%
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

# %%
# Definition of the utility functions with linear cost.
v_train = asc_train_catalog + b_time * train_tt_catalog + b_cost * TRAIN_COST_SCALED
v_swissmetro = b_time * sm_tt_catalog + b_cost * SM_COST_SCALED
v_car = asc_car_catalog + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

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
the_biogeme: BIOGEME = BIOGEME(database=database, formulas=log_probability)
the_biogeme.model_name = 'b20multiple_models'

# %%
dict_of_results = the_biogeme.estimate_catalog()

# %%
print(f'A total of {len(dict_of_results)} models have been estimated:')
for config, res in dict_of_results.items():
    print(f'{config}: LL={res.final_log_likelihood:.2f} K={res.number_of_parameters}')

# %%
summary, description = compile_estimation_results(dict_of_results, use_short_names=True)
print(summary)

# %%
# Explanation of the names of the models.
for k, v in description.items():
    if k != v:
        print(f'{k}: {v}')

# %%
non_dominated_models = pareto_optimal(dict_of_results)
print(f'Out of them, {len(non_dominated_models)} are non dominated.')
for config, res in non_dominated_models.items():
    print(f'{config}')

# %%
summary, description = compile_estimation_results(
    non_dominated_models, use_short_names=False
)
print(summary)

# %%
# It is possible to generate a LaTeX table comparing the results
latex_code = compare_parameters(estimation_results=dict_of_results)
print(latex_code)
