"""

Factor analysis
===============

Preliminary analysis of the indicators using factor analysis

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 16:39:54 2023
"""

import sys
import pandas as pd
import numpy as np
from IPython.core.display_functions import display
from icecream import ic

from biogeme.data.optima import data_file_path

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
    'Envir01',
    'Envir02',
    'Envir03',
    'Envir04',
    'Envir05',
    'Envir06',
    'Mobil01',
    'Mobil02',
    'Mobil03',
    'Mobil04',
    'Mobil05',
    'Mobil06',
    'Mobil07',
    'Mobil08',
    'Mobil09',
    'Mobil10',
    'Mobil11',
    'Mobil12',
    'Mobil13',
    'Mobil14',
    'Mobil15',
    'Mobil16',
    'Mobil17',
    'Mobil18',
    'Mobil19',
    'Mobil20',
    'Mobil21',
    'Mobil22',
    'Mobil23',
    'Mobil24',
    'Mobil25',
    'Mobil26',
    'Mobil27',
    'ResidCh01',
    'ResidCh02',
    'ResidCh03',
    'ResidCh04',
    'ResidCh05',
    'ResidCh06',
    'ResidCh07',
    'LifSty01',
    'LifSty02',
    'LifSty03',
    'LifSty04',
    'LifSty05',
    'LifSty06',
    'LifSty07',
    'LifSty08',
    'LifSty09',
    'LifSty10',
    'LifSty11',
    'LifSty12',
    'LifSty13',
    'LifSty14',
]

indicators = pd.read_csv(data_file_path, sep='\t', usecols=variables)

# %%
# Negative values are missing values.
indicators[indicators <= 0] = np.nan
indicators = indicators.dropna(axis=0, how='any')

# %%
# We perform the factor analysis
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(indicators)


# Get factor loadings
loadings = fa.loadings_
ic(loadings)
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
