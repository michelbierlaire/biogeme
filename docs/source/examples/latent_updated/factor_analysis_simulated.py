"""

Factor analysis
===============

"""

import sys
import pandas as pd
import numpy as np
from IPython.core.display_functions import display
from icecream import ic
from sklearn.preprocessing import StandardScaler

# %%
# The following package can be installed using
#
# `pip install factor_analyzer`
#
# See https://github.com/EducationalTestingService/factor_analyzer
try:
    from factor_analyzer import FactorAnalyzer
except ModuleNotFoundError:
    print('Use "pip install factor_analyzer" to install a requested package')
    sys.exit()

# %%
# We first extract the columns containing the indicators
variables = [
    'I_1',
    'I_2',
    'I_3',
    'I_4',
    'I_5',
]

data_file_path = 'simulated_data.dat'
indicators = pd.read_csv(data_file_path, usecols=variables)

# %%
# We scale the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(indicators)

# %%
# We perform the factor analysis
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(normalized_data)
factor_loadings = fa.loadings_


# Get factor loadings
loadings = fa.loadings_
loadings_cov = np.dot(factor_loadings.T, factor_loadings)

ic(loadings)

new_data_file_path = 'another_simulated_data.dat'
new_data = pd.read_csv(data_file_path, usecols=variables)
new_normalized_data = scaler.transform(new_data)
ic(new_normalized_data.shape)

factor_scores = np.dot(
    new_normalized_data, np.dot(factor_loadings, np.linalg.inv(loadings_cov))
)
ic(factor_scores)

sys.exit()
# Pseudo-inverse
ic(np.linalg.pinv(loadings))

# Get factor scores (these are the reconstructed factors)
factor_scores = fa.transform(indicators)

# Variance of the error terms
unique_variances = fa.get_uniquenesses()

# Convert to a DataFrame for easier visualization
factor_scores_df = pd.DataFrame(factor_scores, columns=['Factor_1', 'Factor_2'])

# Show the reconstructed factors

display(factor_scores_df)

ic(indicators @ np.linalg.pinv(loadings).T)
# Pseudo inverse factor scores
pinv_factor_scores_df = pd.DataFrame(
    indicators @ np.linalg.pinv(loadings).T, columns=['Factor_1', 'Factor_2']
)
display(pinv_factor_scores_df)
# Get uniqueness (related to the error term)
uniqueness = fa.get_uniquenesses()
print('Uniqueness (error term):', uniqueness)
