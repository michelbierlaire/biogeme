"""

MIMIC (Multiple Indicators Multiple Causes) model
=================================================

THe MIMIC model involves two latent_old variables: "car centric" attitude, and "urban preference" attitude.
We consider the indicators as continuous
Michel Bierlaire, EPFL
Fri May 16 2025, 10:32:30

"""

from IPython.core.display_functions import display
from measurement_equations_likert import generate_likert_measurement_equations
from optima import read_data
from read_or_estimate import read_or_estimate
from structural_equations import (
    LatentVariable,
    build_car_centric_attitude,
    build_urban_preference_attitude,
)

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Expression, MonteCarlo, log
from biogeme.results_processing import (
    get_pandas_estimated_parameters,
)

logger = blog.get_screen_logger(level=blog.INFO)

NUMBER_OF_DRAWS = 10

# %%
# Structural equation: car centric attitude
car_centric_attitude: LatentVariable = build_car_centric_attitude()

# %%
# Structural equation: urban preference
urban_preference_attitude: LatentVariable = build_urban_preference_attitude()

# %%
# Generate the measurement equations
measurement_equations: Expression = generate_likert_measurement_equations(
    car_centric_attitude, urban_preference_attitude
)

# %%
# Generate the loglikelihood function
log_likelihood = log(MonteCarlo(measurement_equations))


# %%
# Read the data
database: Database = read_data()

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=NUMBER_OF_DRAWS,
    calculating_second_derivatives='never',
    numerically_safe=True,
    max_iterations=5000,
)
the_biogeme.model_name = 'b01_mimic_discrete'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results = read_or_estimate(the_biogeme=the_biogeme, directory='saved_results')

# %%
print(f'Estimated betas: {results.number_of_parameters}')
print(f'final log likelihood: {results.final_log_likelihood:.3f}')
print(f'Output file: {the_biogeme.html_filename}')

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
