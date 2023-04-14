"""File runknapsack.py

:author: Michel Bierlaire
:date: Fri Apr 14 14:33:18 2023

Example of how to run the VNS multi-objective optimization algorithm
on the knapsack problem.  The two objectives are: minimizing weight
and maximizing utility.

"""

from itertools import product
from biogeme import vns
import biogeme.logging as blog
from knapsack import Knapsack, Sack

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example runknapsack.py')

UTILITY = [80, 31, 48, 17, 27, 84, 34, 39, 46, 58, 23, 67]
WEIGHT = [84, 27, 47, 22, 21, 96, 42, 46, 54, 53, 32, 78]
CAPACITY = 300
FILE_NAME = 'knapsack.pareto'
Sack.utility_data = UTILITY
Sack.weight_data = WEIGHT

the_pareto = vns.ParetoClass(max_neighborhood=5, pareto_file=FILE_NAME)

the_knapsack = Knapsack(UTILITY, WEIGHT, CAPACITY)

all_combinations = product([0, 1], repeat=len(UTILITY))

total = 0
valid = 0
for decision in all_combinations:
    total += 1
    the_sack = Sack.from_decisions(decision)
    is_valid, why = the_knapsack.is_valid(the_sack.get_element())
    if is_valid:
        valid += 1

print(f'Total number of sacks: {total}')
print(f'Total number of valid sacks: {valid}')

empty_sack = Sack.empty_sack(size=len(UTILITY))

the_pareto = vns.vns(
    problem=the_knapsack,
    first_solutions=[empty_sack.get_element()],
    pareto=the_pareto,
    number_of_neighbors=5,
)

print(f'Number of pareto solutions: {len(the_pareto.pareto)}')
print(f'Number of considered solutions: {len(the_pareto.considered)}')
print(f'Pareto solutions: {the_pareto.pareto}')

for p in the_pareto.pareto:
    the_sack = Sack(p.element_id)
    print(the_sack.describe())
