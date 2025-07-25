"""

Catalog for segmented parameters
================================

Investigate the segmentations of parameters.

We consider 4 specifications for the constants:

    - Not segmented
    - Segmented by GA (yearly subscription to public transport)
    - Segmented by luggage
    - Segmented both by GA and luggage

We consider 3 specifications for the time coefficients:

    - Not Segmented
    - Segmented with first class
    - Segmented with trip purpose

We obtain a total of 12 specifications.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

Michel Bierlaire, EPFL
Sun Apr 27 2025, 15:52:48
"""

import numpy as np
from IPython.core.display_functions import display

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
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import compile_estimation_results, pareto_optimal

# %%
# Read the data
database = read_data()

# %%
# Definition of the segmentations.
segmentation_ga = database.generate_segmentation(
    variable='GA', mapping={0: 'noGA', 1: 'GA'}
)

segmentation_luggage = database.generate_segmentation(
    variable='LUGGAGE', mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'}
)

segmentation_first = database.generate_segmentation(
    variable='FIRST', mapping={0: '2nd_class', 1: '1st_class'}
)


# %%
# We consider two trip purposes: 'commuters' and anything else. We
# need to define a binary variable first.
database.dataframe['COMMUTERS'] = np.where(database.dataframe['PURPOSE'] == 1, 1, 0)

segmentation_purpose = database.generate_segmentation(
    variable='COMMUTERS', mapping={0: 'non_commuters', 1: 'commuters'}
)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Catalogs for the alternative specific constants.
asc_train_catalog, asc_car_catalog = segmentation_catalogs(
    generic_name='asc',
    beta_parameters=[asc_train, asc_car],
    potential_segmentations=(
        segmentation_ga,
        segmentation_luggage,
    ),
    maximum_number=2,
)

# %%
# Catalog for the travel time coefficient.
# Note that the function returns a list of catalogs. Here, the list
# contains only one of them.  This is why there is a comma after
# "B_TIME_catalog".
(b_time_catalog,) = segmentation_catalogs(
    generic_name='b_time',
    beta_parameters=[b_time],
    potential_segmentations=(
        segmentation_first,
        segmentation_purpose,
    ),
    maximum_number=1,
)

# %%
# Definition of the utility functions.
v_train = (
    asc_train_catalog + b_time_catalog * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
)
v_swissmetro = b_time_catalog * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car_catalog + b_time_catalog * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

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
# Create the Biogeme object.
the_biogeme = BIOGEME(
    database, log_probability, generate_html=False, generate_yaml=False
)
the_biogeme.model_name = 'b04segmentation'

# %%
# Estimate the parameters
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
