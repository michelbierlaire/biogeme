"""File non_monotonic_forecasting.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 18:36:50 2024

Forecasting with a MDCEV model and the "non monotonic utility" specification.
"""

import pickle
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
one_row_of_database: Database = database.extract_rows([1])

# %
budget_in_hours = 500

# %
number_of_draws = 1

trial = True

if trial:
    epsilons = [
        np.random.gumbel(
            loc=0,
            scale=1,
            size=(number_of_draws, the_non_monotonic.number_of_alternatives),
        )
        for _ in range(one_row_of_database.get_sample_size())
    ]
    with open('epsilons.pkl', 'wb') as file:
        pickle.dump(epsilons, file)
else:

    with open('epsilons.pkl', 'rb') as file:
        epsilons = pickle.load(file)


# %
start_time = time.time()
optimal_consumptions: list[pd.DataFrame] = the_non_monotonic.forecast(
    database=one_row_of_database,
    total_budget=budget_in_hours,
    number_of_draws=number_of_draws,
    user_defined_epsilon=epsilons,
    brute_force=False,
)
end_time = time.time()

# %
print(f'Execution time for {number_of_draws} draws: {end_time-start_time} seconds')

# %
start_time = time.time()
optimal_consumptions_brute_force: list[pd.DataFrame] = the_non_monotonic.forecast(
    database=one_row_of_database,
    total_budget=budget_in_hours,
    number_of_draws=number_of_draws,
    user_defined_epsilon=epsilons,
    brute_force=True,
)
end_time = time.time()

# %
print(f'Execution time for {number_of_draws} draws: {end_time-start_time} seconds')


# %
for index, result in enumerate(optimal_consumptions):
    print(f'===== Obs. {index} ===========')
    print(result.describe())

print('Brute force')
for index, result in enumerate(optimal_consumptions_brute_force):
    print(f'===== Obs. {index} ===========')
    print(result.describe())
