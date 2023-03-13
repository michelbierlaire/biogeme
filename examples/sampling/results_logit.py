""" Analyze the estimation results for the logit model

:author: Michel Bierlaire
:date: Mon Jan  9 13:16:28 2023
"""

from biogeme.results import compileEstimationResults
from constants import N_ALT, INSTANCES, partitions

MODEL = 'logit'

all_results = {}
for p in partitions:
    results = {
        f'{p}_{n}': f'restaurants_{p}_{MODEL}_{N_ALT}_{n}.pickle'
        for n in range(INSTANCES)
    }
    all_results[p] = compileEstimationResults(results, include_robust_stderr=True, formatted=True)

for name, table in all_results.items():
    print('""""""""""""""""""')
    print(name)
    print('""""""""""""""""""')
    print(table)
