"""

Segmentations and alternative specific specification
====================================================

We consider 4 specifications for the constants:

- Not segmented
- Segmented by GA (yearly subscription to public transport)
- Segmented by luggage
- Segmented both by GA and luggage

We consider 6 specifications for the time coefficients:

- Generic and not segmented
- Generic and segmented with first class
- Generic and segmented with trip purpose
- Alternative specific and not segmented
- Alternative specific and segmented with first class
- Alternative specific and segmented with trip purpose

We consider 2 specifications for the cost coefficients:

- Generic
- Alternative specific

We obtain a total of 48 specifications.
See `Bierlaire and Ortelli (2023)
<https://transp-or.epfl.ch/documents/technicalReports/BierOrte23.pdf>`_.

:author: Michel Bierlaire, EPFL
:date: Thu Jul 13 16:18:10 2023

"""
import numpy as np
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from biogeme.catalog import segmentation_catalogs, generic_alt_specific_catalogs
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
database.data['COMMUTERS'] = np.where(database.data['PURPOSE'] == 1, 1, 0)

segmentation_purpose = database.generate_segmentation(
    variable='COMMUTERS', mapping={0: 'non_commuters', 1: 'commuters'}
)

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Catalogs for the alternative specific constants.
ASC_TRAIN_catalog, ASC_CAR_catalog = segmentation_catalogs(
    generic_name='ASC',
    beta_parameters=[ASC_TRAIN, ASC_CAR],
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
(B_TIME_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_TIME',
    beta_parameters=[B_TIME],
    alternatives=['TRAIN', 'SM', 'CAR'],
    potential_segmentations=(
        segmentation_first,
        segmentation_purpose,
    ),
    maximum_number=1,
)

# %%
# Catalog for the travel cost coefficient.
(B_COST_catalog_dict,) = generic_alt_specific_catalogs(
    generic_name='B_COST', beta_parameters=[B_COST], alternatives=['TRAIN', 'SM', 'CAR']
)

# %%
# Definition of the utility functions.
V1 = (
    ASC_TRAIN_catalog
    + B_TIME_catalog_dict['TRAIN'] * TRAIN_TT_SCALED
    + B_COST_catalog_dict['TRAIN'] * TRAIN_COST_SCALED
)
V2 = (
    B_TIME_catalog_dict['SM'] * SM_TT_SCALED
    + B_COST_catalog_dict['SM'] * SM_COST_SCALED
)
V3 = (
    ASC_CAR_catalog
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
the_biogeme.modelName = 'b05alt_spec_segmentation'
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
