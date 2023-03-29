import sys
from itertools import product
from biogeme import vns
import biogeme.messaging as msg
from knapsack import Knapsack, Sack

logger = msg.bioMessage()
logger.setDebug()
# Run one instance

utility = [80, 31, 48, 17, 27, 84, 34, 39, 46, 58, 23, 67]
weight = [84, 27, 47, 22, 21, 96, 42, 46, 54, 53, 32, 78]
CAPACITY = 300
Sack.utility_data = utility
Sack.weight_data = weight

the_knapsack = Knapsack(utility, weight, CAPACITY)

all_combinations = product([0, 1], repeat=len(utility))

total = 0
valid = 0
for decision in all_combinations:
    total += 1
    the_sack = Sack.from_decisions(decision)
    is_valid, why =  the_knapsack.is_valid(the_sack.get_element())
    if is_valid:
        valid += 1

print(f'Total number of sacks: {total}')
print(f'Total number of valid sacks: {valid}')

empty_sack = Sack.empty_sack(size=len(utility))

the_pareto = vns.vns(
    the_knapsack,
    [empty_sack.get_element()],
    max_neighborhood=12,
    number_of_neighbors=10,
    pareto_file_name='knapsack.pareto',
)

print(f'Number of pareto solutions: {len(the_pareto.pareto)}')
print(f'Number of considered solutions: {len(the_pareto.considered)}')
print(f'Pareto solutions: {the_pareto.pareto}')

for p in the_pareto.pareto:
    the_sack = Sack(p.element_id)
    print(the_sack.describe())
