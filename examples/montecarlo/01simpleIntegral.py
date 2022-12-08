"""File: 01simpleIntegral.py
 Author: Michel Bierlaire, EPFL
 Date: Wed Dec 11 16:20:24 2019

Calculation of a simple integral using Monte-Carlo integration.

"""

# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import exp, bioDraws, MonteCarlo
from biogeme.tools import TemporaryFile

# We create a fake database with one entry, as it is required
# to store the draws
pandas = pd.DataFrame()
pandas['FakeColumn'] = [1.0]
database = db.Database('fakeDatabase', pandas)

integrand = exp(bioDraws('U', 'UNIFORM'))
simulatedI = MonteCarlo(integrand)

trueI = exp(1.0) - 1.0

R = 200

# Create a parameter file to set the number of draws
with TemporaryFile() as filename:
    with open(filename, 'w', encoding='utf-8') as f:
        print('[MonteCarlo]', file=f)
        print(f'number_of_draws = {R}', file=f)

    sampleVariance = MonteCarlo(integrand * integrand) - simulatedI * simulatedI
    stderr = (sampleVariance / R) ** 0.5
    error = simulatedI - trueI

    simulate = {
        'Analytical Integral': trueI,
        'Simulated Integral': simulatedI,
        'Sample variance   ': sampleVariance,
        'Std Error         ': stderr,
        'Error             ': error,
    }

    biogeme = bio.BIOGEME(database, simulate, parameter_file=filename)
    R = biogeme.number_of_draws
    biogeme.modelName = f'01simpleIntegral_{R}'
    results = biogeme.simulate()
    print(f'Number of draws: {R}')
    for c in results.columns:
        print(f'{c}: {results.loc[0,c]}')

multiplier = 100000
        
# Create a parameter file to set the umber of draws
with TemporaryFile() as filename:
    with open(filename, 'w', encoding='utf-8') as f:
        print('[MonteCarlo]', file=f)
        print(f'number_of_draws = {multiplier * R}', file=f)


    # With 10 times more draws
    biogeme2 = bio.BIOGEME(database, simulate, parameter_file=filename)
    biogeme2.modelName = '01simpleIntegral_{multiplier*R}'
    results2 = biogeme2.simulate()
    print(f'Number of draws: {multiplier * R}')
    for c in results2.columns:
        print(f'{c}: {results2.loc[0, c]}')
