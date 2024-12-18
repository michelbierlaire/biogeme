"""

Illustration of additional features of Biogeme
==============================================

Same model as b01logit, using bioLinearUtility, segmentations
 and features.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:03:31 2023

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, bioLinearUtility, LinearTermTuple
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters
from biogeme.segmentation import Segmentation

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    database,
    CHOICE,
    GA,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    MALE,
    SM_AV,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01logit_bis.py')

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)

# %%
# Starting value.
# We use starting values estimated from a previous run
B_TIME = Beta('B_TIME', -1.28, None, None, 0)
B_COST = Beta('B_COST', -1.08, None, None, 0)

# %%
# Define segmentations.
gender_segmentation = database.generate_segmentation(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

GA_segmentation = database.generate_segmentation(
    variable=GA, mapping={0: 'without_ga', 1: 'with_ga'}
)

segmentations_for_asc = [
    gender_segmentation,
    GA_segmentation,
]

# %%
# Segmentation of the constants.
ASC_TRAIN_segmentation = Segmentation(ASC_TRAIN, segmentations_for_asc)
segmented_ASC_TRAIN = ASC_TRAIN_segmentation.segmented_beta()
ASC_CAR_segmentation = Segmentation(ASC_CAR, segmentations_for_asc)
segmented_ASC_CAR = ASC_CAR_segmentation.segmented_beta()

# %%
# Definition of the utility functions.
terms1 = [
    LinearTermTuple(beta=B_TIME, x=TRAIN_TT_SCALED),
    LinearTermTuple(beta=B_COST, x=TRAIN_COST_SCALED),
]
V1 = segmented_ASC_TRAIN + bioLinearUtility(terms1)

terms2 = [
    LinearTermTuple(beta=B_TIME, x=SM_TT_SCALED),
    LinearTermTuple(beta=B_COST, x=SM_COST_SCALED),
]
V2 = bioLinearUtility(terms2)

terms3 = [
    LinearTermTuple(beta=B_TIME, x=CAR_TT_SCALED),
    LinearTermTuple(beta=B_COST, x=CAR_CO_SCALED),
]
V3 = segmented_ASC_CAR + bioLinearUtility(terms3)

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
#
# This is the contribution of each observation to the log likelihood
# function.
logprob = loglogit(V, av, CHOICE)

# %%
# User notes.
#
# These notes will be included as such in the report file.
USER_NOTES = (
    'Example of a logit model with three alternatives: Train, Car and'
    ' Swissmetro. Same as 01logit and '
    'introducing some options and features. In particular, bioLinearUtility,'
    ' and automatic segmentation of parameters.'
)


# %%
# Create the Biogeme object. We include users notes, and we ask not to calculate the second derivatives.
the_biogeme = BIOGEME(database, logprob, user_notes=USER_NOTES, second_derivatives=0)

# %%
# Calculate the null log likelihood for reporting.
#
# As we have used starting values different from 0, the initial model
# is not the equal probability model.
the_biogeme.calculate_null_loglikelihood(av)
the_biogeme.modelName = 'b01logit_bis'

# %%
# Turn off saving iterations.
#
the_biogeme.save_iterations = False

# %%
# Estimate the parameters.
#
the_biogeme.bootstrap_samples = 100
results = the_biogeme.estimate(run_bootstrap=True)

# %%
# Get the results in a pandas table.
#
print('Parameters')
print('----------')
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)

# %%
# Get general statistics.
#
print('General statistics')
print('------------------')
stats = results.get_general_statistics()
for description, (value, formatting) in stats.items():
    print(f'{description}: {value:{formatting}}')

# %%
# Messages from the optimization algorithm.
#
print('Optimization algorithm')
print('----------------------')
for description, message in results.data.optimizationMessages.items():
    print(f'{description}:\t{message}')

# %%
# Generate the file in Alogit format.
#
results.write_f12(robust_std_err=True)
print(f'Estimation results in ALogit format generated: {results.data.F12FileName}')

# %%
# Generate LaTeX code with the results.
#
results.write_latex()
print(f'Estimation results in LaTeX format generated: {results.data.latexFileName}')
