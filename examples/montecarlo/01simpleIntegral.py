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

# We create a fake database with one entry, as it is required
# to store the draws
pandas = pd.DataFrame()
pandas['FakeColumn'] = [1.0]
database = db.Database('fakeDatabase', pandas)

integrand = exp(bioDraws('U', 'UNIFORM'))
simulatedI = MonteCarlo(integrand)

trueI = exp(1.0) - 1.0

R = 2000

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

biogeme = bio.BIOGEME(database, simulate, numberOfDraws=R)
biogeme.modelName = f'01simpleIntegral_{R}'
results = biogeme.simulate()
print(f'Number of draws: {R}')
for c in results.columns:
    print(f'{c}: {results.loc[0,c]}')

# With 10 times more draws
biogeme2 = bio.BIOGEME(database, simulate, numberOfDraws=10 * R)
biogeme2.modelName = '01simpleIntegral_{10*R}'
results2 = biogeme2.simulate()
print(f'Number of draws: {10 * R}')
for c in results.columns:
    print(f'{c}: {results2.loc[0, c]}')
