"""File non_monotonic_forecasting.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 18:36:50 2024

Forecasting with a MDCEV model and the "non monotonic utility" specification.
"""

import sys
import time

import numpy as np
import pandas as pd

import biogeme.biogeme_logging as blog
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.results import bioResults
from specification import (
    database,
)
from non_monotonic_specification import the_non_monotonic

logger = blog.get_screen_logger(level=blog.DEBUG)
logger.info('Example: non monotonic utility')

result_file = 'saved_results/non_monotonic.pickle'
try:
    results = bioResults(pickle_file=result_file)
except BiogemeError as e:
    print(e)
    print(f'File {result_file} is missing.')
    sys.exit()

the_non_monotonic.estimation_results = results

# %
# We apply the model only on the first two rows of the database.
two_rows_of_database: Database = database.extract_rows([10, 11])

# %
budget_in_hours = 500

# %
number_of_draws = 20000

# %
start_time = time.time()
optimal_consumptions: list[pd.DataFrame] = the_non_monotonic.forecast(
    database=two_rows_of_database,
    total_budget=budget_in_hours,
    number_of_draws=number_of_draws,
    brute_force=False,
)
end_time = time.time()

# %
print(f'Execution time for {number_of_draws} draws: {end_time-start_time} seconds')

# %
for index, result in enumerate(optimal_consumptions):
    print(f'===== Obs. {index} ===========')
    print(result.describe())

# %
# We now compare the two forecasting algorithms: the brute force, that ignores the properties of the model,
# and the analytical algorithm, based on a bisection method.

# %
# We reduce the number of draws for the sake of illustration
number_of_draws = 200

# %
# To compare the results, we use the same set of draws for both
epsilons = [
    np.random.gumbel(
        loc=0, scale=1, size=(number_of_draws, the_non_monotonic.number_of_alternatives)
    )
    for _ in range(two_rows_of_database.get_sample_size())
]


# %
# First, the brute force algorithm.
start_time = time.time()
optimal_consumptions_brute_force: list[pd.DataFrame] = the_non_monotonic.forecast(
    database=two_rows_of_database,
    total_budget=budget_in_hours,
    number_of_draws=number_of_draws,
    user_defined_epsilon=epsilons,
    brute_force=True,
)
end_time = time.time()

# %
print(
    f'Execution time for {number_of_draws} draws with brute force algorithm: {end_time-start_time} seconds'
)

# %
# Then, the analytical algorithm.
start_time = time.time()
optimal_consumptions_analytical: list[pd.DataFrame] = the_non_monotonic.forecast(
    database=two_rows_of_database,
    total_budget=budget_in_hours,
    number_of_draws=number_of_draws,
    user_defined_epsilon=epsilons,
    brute_force=False,
)
end_time = time.time()

# %
print(
    f'Execution time for {number_of_draws} draws with analytical algorithm: {end_time-start_time} seconds'
)

# %
# Results for the first observation, brute force method
optimal_consumptions_brute_force[0].describe()

# %
# Results for the first observation, analytical method
optimal_consumptions_analytical[0].describe()

# %
# Results for the second observation, brute force method
optimal_consumptions_brute_force[1].describe()

# %
# Results for the second observation, analytical method
optimal_consumptions_analytical[1].describe()
