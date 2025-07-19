"""

MIMIC (Multiple Indicators Multiple Causes) model
=================================================

THe MIMIC model involves two latent variables: "car centric" attitude, and "urban preference" attitude.
Michel Bierlaire, EPFL
Fri May 16 2025, 10:32:30

"""

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.data.optima import (
    read_data,
)
from biogeme.database import Database
from biogeme.expressions import (
    Elem,
    MultipleSum,
    Variable,
    log,
)
from biogeme.results_processing import (
    get_pandas_estimated_parameters,
)

from measurement_equations import all_indicators, generate_measurement_equations
from read_or_estimate import read_or_estimate
from structural_equations import (
    build_car_centric_attitude,
    build_urban_preference_attitude,
)

# %%
# Structural equation: car centric attitude
car_centric_attitude = build_car_centric_attitude()

# %%
# Structural equation: urban preference
urban_preference_attitude = build_urban_preference_attitude()

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01one_latent_regression.py')

dict_prob_indicators = generate_measurement_equations(
    car_centric_attitude=car_centric_attitude,
    urban_preference_attitude=urban_preference_attitude,
)

# %%
# We calculate the joint probability of all indicators
log_proba = {
    indicator: log(Elem(dict_prob_indicators[indicator], Variable(indicator)))
    for indicator in all_indicators
}
log_likelihood = MultipleSum(log_proba)

# %%
# Read the data
database: Database = read_data()

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_likelihood)
the_biogeme.model_name = 'b01_mimic'

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
