"""

Read pickle files
=================

Read pickle files with estimation results from previous versions of Biogeme

Michel Bierlaire, EPFL
Sat Oct 5 14:24:39 2024
"""

from biogeme.results_processing import EstimationResults, pickle_to_yaml

results_3_2_7 = EstimationResults.from_pickle_file(filename='b01logit_3_2_7.pickle')
print(f'Results from Biogeme 3.2.7: {results_3_2_7}')

results_3_2_8 = EstimationResults.from_pickle_file(filename='b01logit_3_2_8.pickle')
print(f'Results from Biogeme 3.2.8: {results_3_2_8}')

results_3_2_10 = EstimationResults.from_pickle_file(filename='b01logit_3_2_10.pickle')
print(f'Results from Biogeme 3.2.10: {results_3_2_10}')

results_3_2_11 = EstimationResults.from_pickle_file(filename='b01logit_3_2_11.pickle')
print(f'Results from Biogeme 3.2.11: {results_3_2_11}')

results_3_2_12 = EstimationResults.from_pickle_file(filename='b01logit_3_2_12.pickle')
print(f'Results from Biogeme 3.2.12: {results_3_2_12}')

results_3_2_13 = EstimationResults.from_pickle_file(filename='b01logit_3_2_13.pickle')
print(f'Results from Biogeme 3.2.13: {results_3_2_13}')

results_3_2_14 = EstimationResults.from_pickle_file(filename='b01logit_3_2_14.pickle')
print(f'Results from Biogeme 3.2.14: {results_3_2_14}')

pickle_to_yaml(pickle_filename='b01logit_3_2_7.pickle', yaml_filename='b01logit.yaml')
