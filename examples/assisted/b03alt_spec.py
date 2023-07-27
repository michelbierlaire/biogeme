"""File b03alt_spec.py

:author: Michel Bierlaire, EPFL
:date: Thu Jul 13 16:18:10 2023

Investigate alternative specific parameters:
- two specifications for the travel time coefficient: generic, and alternative specific,
- two specifications for the travel cost coefficient: generic, and alternative specific,
for a total of 4 specifications.
"""
import numpy as np
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from biogeme.catalog import generic_alt_specific_catalogs

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

segmentation_ga = database.generate_segmentation(
    variable='GA', mapping={0: 'noGA', 1: 'GA'}
)

segmentation_luggage = database.generate_segmentation(
    variable='LUGGAGE', mapping={0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'}
)

segmentation_first = database.generate_segmentation(
    variable='FIRST', mapping={0: '2nd_class', 1: '1st_class'}
)


# We consider two trip purposes: 'commuters' and anything else. We
# need to define a binary variable first

database.data['COMMUTERS'] = np.where(database.data['PURPOSE'] == 1, 1, 0)

segmentation_purpose = database.generate_segmentation(
    variable='COMMUTERS', mapping={0: 'non_commuters', 1: 'commuters'}
)


# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

(B_TIME_catalog,) = generic_alt_specific_catalogs(
    generic_name='B_TIME', beta_parameters=[B_TIME], alternatives=('TRAIN', 'SM', 'CAR')
)

(B_COST_catalog,) = generic_alt_specific_catalogs(
    generic_name='B_COST', beta_parameters=[B_COST], alternatives=('TRAIN', 'SM', 'CAR')
)

# Definition of the utility functions
V1 = (
    ASC_TRAIN
    + B_TIME_catalog['TRAIN'] * TRAIN_TT_SCALED
    + B_COST_catalog['TRAIN'] * TRAIN_COST_SCALED
)
V2 = B_TIME_catalog['SM'] * SM_TT_SCALED + B_COST_catalog['SM'] * SM_COST_SCALED
V3 = (
    ASC_CAR
    + B_TIME_catalog['CAR'] * CAR_TT_SCALED
    + B_COST_catalog['CAR'] * CAR_CO_SCALED
)

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b01alt_spec'
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False

# Estimate the parameters
dict_of_results = the_biogeme.estimate_catalog()

report(dict_of_results)
