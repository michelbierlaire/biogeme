"""

Illustration of additional features of Biogeme
==============================================

Same model as b01logit, using LinearUtility, segmentations

Michel Bierlaire, EPFL
Wed Jun 18 2025, 10:57:53

"""

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, LinearTermTuple, LinearUtility
from biogeme.models import loglogit
from biogeme.results_processing import (
    EstimateVarianceCovariance,
    generate_html_file,
    get_pandas_estimated_parameters,
)
from biogeme.segmentation import Segmentation

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    MALE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01logit_bis.py')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)

# %%
# Starting value.
# We use starting values estimated from a previous run
b_time = Beta('b_time', -1.28, None, None, 0)
b_cost = Beta('b_cost', -1.08, None, None, 0)

# %%
# Define segmentations.
gender_segmentation = database.generate_segmentation(
    variable=MALE, mapping={0: 'female', 1: 'male'}
)

ga_segmentation = database.generate_segmentation(
    variable=GA, mapping={0: 'without_ga', 1: 'with_ga'}
)

segmentations_for_asc = [
    gender_segmentation,
    ga_segmentation,
]

# %%
# Segmentation of the constants.
asc_train_segmentation = Segmentation(asc_train, segmentations_for_asc)
segmented_asc_train = asc_train_segmentation.segmented_beta()
asc_car_segmentation = Segmentation(asc_car, segmentations_for_asc)
segmented_asc_car = asc_car_segmentation.segmented_beta()

# %%
# Definition of the utility functions.
terms1 = [
    LinearTermTuple(beta=b_time, x=TRAIN_TT_SCALED),
    LinearTermTuple(beta=b_cost, x=TRAIN_COST_SCALED),
]
v_train = segmented_asc_train + LinearUtility(terms1)

terms2 = [
    LinearTermTuple(beta=b_time, x=SM_TT_SCALED),
    LinearTermTuple(beta=b_cost, x=SM_COST_SCALED),
]
v_swissmetro = LinearUtility(terms2)

terms3 = [
    LinearTermTuple(beta=b_time, x=CAR_TT_SCALED),
    LinearTermTuple(beta=b_cost, x=CAR_CO_SCALED),
]
v_car = segmented_asc_car + LinearUtility(terms3)

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
#
# This is the contribution of each observation to the log likelihood
# function.
logprob = loglogit(v, av, CHOICE)

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
# The parameter 'calculating_second_derivatives' is a general instruction for Biogeme, In this case, the
# second derivatives will not even be calculated after the algorithm has converged. It means that the statistics
# will have to rely on bootstrap or BHHH.
the_biogeme = BIOGEME(
    database,
    logprob,
    user_notes=USER_NOTES,
    save_iterations=False,
    bootstrap_samples=100,
    calculating_second_derivatives='never',
)

# %%
# Calculate the null log likelihood for reporting.
#
# As we have used starting values different from 0, the initial model
# is not the equal probability model.
the_biogeme.calculate_null_loglikelihood(av)
the_biogeme.model_name = 'b01logit_bis'

# %%
# Estimate the parameters.
#
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
for description, value in stats.items():
    print(f'{description}: {value}')

# %%
# Messages from the optimization algorithm.
#
print('Optimization algorithm')
print('----------------------')
for description, message in results.optimization_messages.items():
    print(f'{description}:\t{message}')

# %%
# Try to generate the html output with the robust variance-covariance matrix. It does not work as the second derivatives
# matrix is not calculated.
try:
    robust_html_filename = f'{the_biogeme.model_name}_robust.html'
    generate_html_file(
        filename=robust_html_filename,
        estimation_results=results,
        variance_covariance_type=EstimateVarianceCovariance.ROBUST,
    )
    print(
        f'Estimation results with robust statistics generated: {robust_html_filename}'
    )
except BiogemeError as e:
    print(f'BiogemeError: {e}')

# %%
# Generate the html output with the BHHH variance-covariance matrix
bhhh_html_filename = f'{the_biogeme.model_name}_bhhh.html'
generate_html_file(
    filename=bhhh_html_filename,
    estimation_results=results,
    variance_covariance_type=EstimateVarianceCovariance.BHHH,
)
print(f'Estimation results with BHHH statistics generated: {bhhh_html_filename}')

# %%
# Generate the file in Alogit format.
#
f12_filename = results.write_f12()
print(f'Estimation results in ALogit format generated: {f12_filename}')

# %%
# Generate LaTeX code with the results.
#
latex_filename = results.write_latex(include_begin_document=True)
print(f'Estimation results in LaTeX format generated: {latex_filename}')
