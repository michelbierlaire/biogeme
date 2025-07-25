"""File translated_forecasting.py

Michel Bierlaire, EPFL
Fri Jul 25 2025, 17:34:35

Forecasting with a MDCEV model and the "translated utility" specification.
"""

import sys
import time

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.database import Database
from biogeme.results_processing import EstimationResults
from process_data import database
from translated_specification import the_translated

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: translated utility')

result_file = 'saved_results/translated.yaml'
try:
    results = EstimationResults.from_yaml_file(filename=result_file)
except FileNotFoundError as e:
    print(e)
    print(f'File {result_file} is missing.')
    sys.exit()

the_translated.estimation_results = results

# %
# We apply the model only on the first two rows of the database.
two_rows_of_database: Database = database.extract_rows([0, 1])

# %
budget_in_hours = 500

# %
# # Validation

# %
# As the implementation is still experimental, we compare the result obtained by the bruteforce algorithm and
# the analytical algorithm for a few draws.

# Note that minor discrepancies between the outcome of the two algorithms are likely to occur, due to numerical
# imprecision, inevitable in finite arithmetic.

# However, if there are major differences, it should be reported.

# %
number_of_draws = 10

# %
# We generate the draws
epsilons = [
    np.random.gumbel(
        loc=0, scale=1, size=(number_of_draws, the_translated.number_of_alternatives)
    )
    for _ in range(two_rows_of_database.num_rows())
]

# %
# We first compare the results obtained from the brute force and the analytical algorithms, for each draw.
the_translated.validate_forecast(
    database=two_rows_of_database, total_budget=budget_in_hours, epsilons=epsilons
)

# %
# # Forecasting
# We use a larger number of draws to obtain the forecast.

# %
number_of_draws = 2000

# %
# We generate the draws
epsilons = [
    np.random.gumbel(
        loc=0, scale=1, size=(number_of_draws, the_translated.number_of_alternatives)
    )
    for _ in range(two_rows_of_database.num_rows())
]

# %
# First, the brute force algorithm.
start_time = time.time()
optimal_consumptions_brute_force: list[pd.DataFrame] = the_translated.forecast(
    database=two_rows_of_database,
    total_budget=budget_in_hours,
    epsilons=epsilons,
    brute_force=True,
)
end_time = time.time()

# %
print(
    f'Execution time for {number_of_draws} draws with brute force algorithm: {end_time-start_time:.3g} seconds'
)

# %
# Then, the analytical algorithm.
start_time = time.time()
optimal_consumptions_analytical: list[pd.DataFrame] = the_translated.forecast(
    database=two_rows_of_database,
    total_budget=budget_in_hours,
    epsilons=epsilons,
    brute_force=False,
)
end_time = time.time()

# %
print(
    f'Execution time for {number_of_draws} draws with analytical algorithm: {end_time-start_time:.3g} seconds'
)

# %
# Results for the first observation, brute force method
display(optimal_consumptions_brute_force[0].describe())

# %
# Results for the first observation, analytical method
display(optimal_consumptions_analytical[0].describe())

# %
# Results for the second observation, brute force method
display(optimal_consumptions_brute_force[1].describe())

# %
# Results for the second observation, analytical method
display(optimal_consumptions_analytical[1].describe())
