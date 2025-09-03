"""

Estimation of the hybrid choice model
=====================================

Full information estimation of the model combining observed choice and Likert
scale psychometric indicators.

Michel Bierlaire, EPFL
Wed Sept 03 2025, 08:19:40

"""

from choice_model import v
from IPython.core.display_functions import display
from measurement_equations_likert import generate_likert_measurement_equations
from optima import (
    Choice,
    read_data,
)
from read_or_estimate import read_or_estimate
from structural_equations import (
    LatentVariable,
    build_car_centric_attitude,
    build_urban_preference_attitude,
)

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Expression, MonteCarlo, log
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

NUMBER_OF_DRAWS = 10_000

logger = blog.get_screen_logger(level=blog.INFO)

database = read_data()

# %%
# Structural equation: car centric attitude
car_centric_attitude: LatentVariable = build_car_centric_attitude()

# %%
# Structural equation: urban preference
urban_preference_attitude: LatentVariable = build_urban_preference_attitude()

# %%
# Generate the measurement equations for the indicators
measurement_equations: Expression = generate_likert_measurement_equations(
    car_centric_attitude, urban_preference_attitude
)

# %%
# Conditional likelihood for the choice model
choice_likelihood = logit(v, None, Choice)

# %% Conditional likelihood
conditional_likelihood = choice_likelihood * measurement_equations

# %%
# Log likelihood
log_likelihood = log(MonteCarlo(conditional_likelihood))

# %%
# Create the Biogeme object
print('Create the biogeme object')
the_biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=NUMBER_OF_DRAWS,
    calculating_second_derivatives='never',
    numerically_safe=True,
    max_iterations=5000,
)
the_biogeme.model_name = 'b03_hybrid'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results: EstimationResults = read_or_estimate(
    the_biogeme=the_biogeme, directory='saved_results'
)

# %%
print(f'Estimated betas: {results.number_of_parameters}')
print(f'final log likelihood: {results.final_log_likelihood:.3f}')
print(f'Output file: {the_biogeme.html_filename}')

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
