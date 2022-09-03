import numpy as np
from biogeme import vns
from knapsack import knapsack, oneSack

### Run one instance

utility = np.array([80, 31, 48, 17, 27, 84, 34, 39, 46, 58, 23, 67])
weight = np.array([84, 27, 47, 22, 21, 96, 42, 46, 54, 53, 32, 78])
capacity = 300

theKnapsack = knapsack(utility, weight, capacity)
emptySack = theKnapsack.emptySack()

thePareto = vns.vns(
    theKnapsack,
    [emptySack],
    maxNeighborhood=12,
    numberOfNeighbors=10,
    archiveInputFile='knapsackPareto.pickle',
    pickleOutputFile='knapsackPareto.pickle',
)

print(f'Number of pareto solutions: {len(thePareto.pareto)}')
print(f'Pareto solutions: {thePareto.pareto}')

for p in thePareto.pareto.keys():
    obj = [f'{t}: {r} ' for t, r in zip(p.objectivesNames, p.objectives)]
    print(f'{p} {obj}')
