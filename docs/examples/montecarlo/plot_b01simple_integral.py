"""

Simple integral
===============

Calculation of a simple integral using Monte-Carlo integration.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 20:42:24 2023

"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import exp, bioDraws, MonteCarlo
from biogeme.tools import TemporaryFile

# %%
# We create a fake database with one entry, as it is required
# to store the draws
df = pd.DataFrame()
df['FakeColumn'] = [1.0]
database = db.Database('fakeDatabase', df)

# %%
integrand = exp(bioDraws('U', 'UNIFORM'))
simulatedI = MonteCarlo(integrand)

# %%
trueI = exp(1.0) - 1.0

# %%
R = 200
MULTIPLIER = 100000

# %%
# Create a parameter file to set the number of draws.
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

    biosim = bio.BIOGEME(database, simulate, parameter_file=filename)
    R = biosim.number_of_draws
    biosim.modelName = f'01simpleIntegral_{R}'
    results = biosim.simulate(theBetaValues={})
    print(f'Number of draws: {R}')
    for c in results.columns:
        print(f'{c}: {results.loc[0,c]}')

# %%
# Create a parameter file to set the number of draws.
with TemporaryFile() as filename:
    with open(filename, 'w', encoding='utf-8') as f:
        print('[MonteCarlo]', file=f)
        print(f'number_of_draws = {MULTIPLIER * R}', file=f)

    biogeme2 = bio.BIOGEME(database, simulate, parameter_file=filename)
    biogeme2.modelName = '01simpleIntegral_{multiplier*R}'
    results2 = biogeme2.simulate(theBetaValues={})
    print(f'Number of draws: {MULTIPLIER * R}')
    for c in results2.columns:
        print(f'{c}: {results2.loc[0, c]}')
