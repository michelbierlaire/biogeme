""" Analyze the estimation results for the cross-nested logit model

:author: Michel Bierlaire
:date: Tue Jan 10 15:14:36 2023
"""

from biogeme.results import compileEstimationResults
from constants import N_ALT, INSTANCES, partitions

MODEL = 'cnl'

all_results = {}
for p in partitions:
    results = {
        f'{p}_{n}': f'restaurants_{p}_{MODEL}_{N_ALT}_{n}.pickle'
        for n in range(INSTANCES)
    }
    all_results[p] = compileEstimationResults(results)

print(all_results)
