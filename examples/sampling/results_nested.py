""" Analyze the estimation results for the nested logit model

:author: Michel Bierlaire
:date: Mon Jan  9 16:56:31 2023
"""

from biogeme.results import compileEstimationResults
from constants import N_ALT, INSTANCES, partitions

MODEL = 'nested'

all_results = {}
for p in partitions:
    results = {
        f'{p}_{n}': f'restaurants_{p}_{MODEL}_{N_ALT}_{n}.pickle'
        for n in range(INSTANCES)
    }
    all_results[p] = compileEstimationResults(results)

print(all_results)
