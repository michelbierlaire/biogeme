"""

1b. Illustration of additional Biogeme features
===============================================

This example estimates the same logit model as in Example 1a, but
illustrates several additional features available in Biogeme.

In particular, it demonstrates:

- the use of `LinearUtility` to define utility functions,
- automatic parameter segmentation,
- the generation of alternative variance-covariance matrices,
- the production of several output formats.

The model considers three transportation alternatives:

- Train,
- Swissmetro,
- Car.

The utility functions include alternative-specific constants and
generic coefficients associated with travel time and travel cost.
The Swissmetro alternative is used as the reference alternative.

The `# %%` markers are used to separate the script into notebook
cells when the example gallery is converted into Jupyter notebooks.

Tested with Biogeme 3.3.3.

Michel Bierlaire, EPFL
Tue Jun 09 2026, 14:40:00

"""

import os

from IPython.core.display_functions import display

# %%
# Import the variables and the database prepared in the Swissmetro data-processing example.
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

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, LinearTermTuple, LinearUtility
from biogeme.models import loglogit
from biogeme.results_processing import (
    EstimateVarianceCovariance,
    EstimationResults,
    generate_html_file,
    get_pandas_estimated_parameters,
)
from biogeme.segmentation import Segmentation

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01logit_bis.py')

# %%
# Define the model parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)

# %%
# Starting values obtained from a previous estimation run.
b_time = Beta('b_time', -1.28, None, None, 0)
b_cost = Beta('b_cost', -1.08, None, None, 0)

# %%
# Define the segmentation schemes used for the alternative-specific constants.
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
# Apply the segmentations to the alternative-specific constants.
asc_train_segmentation = Segmentation(asc_train, segmentations_for_asc)
segmented_asc_train = asc_train_segmentation.segmented_beta()
asc_car_segmentation = Segmentation(asc_car, segmentations_for_asc)
segmented_asc_car = asc_car_segmentation.segmented_beta()

#
# Define the utility functions. A `LinearTermTuple` combines a coefficient
# and an explanatory variable. A `LinearUtility` is the sum of the
# products of each coefficient by its associated variable.
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
# Associate each utility function with the corresponding alternative identifier.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with each alternative.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Define the log-likelihood contribution of each observation.
logprob = loglogit(v, av, CHOICE)

# %%
# User notes that will be included in the generated report.
USER_NOTES = (
    'Example of a logit model with three alternatives: Train, Car and'
    ' Swissmetro. Same as 01logit and '
    'introducing some options and features. In particular, LinearUtility,'
    ' and automatic segmentation of parameters.'
)


# %%
# Create the Biogeme object. Second derivatives are disabled. Therefore,
# statistics requiring the Hessian matrix will not be available and
# alternative procedures such as bootstrap or BHHH must be used.
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
the_biogeme.model_name = 'b01b_logit'

# %%
# Estimate the parameters or retrieve previously saved results.
try:
    results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{the_biogeme.model_name}.yaml'
    )
except FileNotFoundError:
    results = the_biogeme.estimate(run_bootstrap=True)

# %%
# Convert the estimated parameters into a pandas DataFrame.
print('Parameters')
print('----------')
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)

# %%
# Display general estimation statistics.
print('General statistics')
print('------------------')
stats = results.get_general_statistics()
for description, value in stats.items():
    print(f'{description}: {value}')

# %%
# Display messages returned by the optimization algorithm.
print('Optimization algorithm')
print('----------------------')
for description, message in results.optimization_messages.items():
    print(f'{description}:\t{message}')

# %%
# Attempt to generate an HTML report based on the robust
# variance-covariance matrix. This fails because second derivatives
# have not been calculated.
try:
    robust_html_filename = f'{the_biogeme.model_name}_robust.html'
    # The following function assumes that the file does not exist.
    if os.path.exists(robust_html_filename):
        os.remove(robust_html_filename)
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
# Generate an HTML report using the BHHH variance-covariance matrix.
bhhh_html_filename = f'{the_biogeme.model_name}_bhhh.html'
# The following function assumes that the file does not exist. Therefore, if it does exist, we erase it.
if os.path.exists(bhhh_html_filename):
    os.remove(bhhh_html_filename)
generate_html_file(
    filename=bhhh_html_filename,
    estimation_results=results,
    variance_covariance_type=EstimateVarianceCovariance.BHHH,
)
print(f'Estimation results with BHHH statistics generated: {bhhh_html_filename}')

# %%
# Generate the results file in ALogit format.
f12_filename = results.write_f12()
print(f'Estimation results in ALogit format generated: {f12_filename}')

# %%
# Generate LaTeX output containing the estimation results.
latex_filename = results.write_latex(include_begin_document=True)
print(f'Estimation results in LaTeX format generated: {latex_filename}')
