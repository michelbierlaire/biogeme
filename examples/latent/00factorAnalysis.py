"""File 00factorAnalysis.py

Preliminary analysis of the indicators using factor analysis.

:author: Michel Bierlaire, EPFL
:date: Mon Sep  9 16:04:57 2019

"""

import sys
import pandas as pd
import numpy as np

# The following package can be installed using
# pip install factor_analyzer
# See https://github.com/EducationalTestingService/factor_analyzer
try:
    from factor_analyzer import FactorAnalyzer
except ModuleNotFoundError:
    print('Use "pip install factor_analyzer" to install a requested package')
    sys.exit()

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

indicators = pd.read_csv('optima.dat', sep='\t', usecols=variables)

# Negative values are missing values.
indicators[indicators <= 0] = np.nan
indicators = indicators.dropna(axis=0, how='any')

# We perform the factor analysis
fa = FactorAnalyzer(rotation='varimax')
fa.fit(indicators)

# We obtain the factor loadings and label them
labeledResults = pd.DataFrame(fa.loadings_)
labeledResults.index = variables

# We keep only loadings that are 0.4 or higher, in absolute value.
labeledResults[(labeledResults <= 0.4) & (labeledResults >= -0.4)] = ''
print(labeledResults)
