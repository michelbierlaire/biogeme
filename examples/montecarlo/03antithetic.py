"""File: 03antithetic.py

 Author: Michel Bierlaire, EPFL
 Date: Wed Dec 11 17:00:05 2019

Calculation of a simple integral using Monte-Carlo integration. It
illustrates how to use antithetic draws.
"""

# pylint: disable=invalid-name, undefined-variable
import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import draws
from biogeme.expressions import exp, bioDraws, MonteCarlo


# We create a fake database with one entry, as it is required to store
# the draws
pandas = pd.DataFrame()
pandas['FakeColumn'] = [1.0]
database = db.Database('fakeDatabase', pandas)


def halton13_anti(sampleSize, numberOfDraws):
    """The user can define new draws. For example, Halton draws with base
    13, skipping the first 10 draws.

    """
    d = draws.getHaltonDraws(
        sampleSize, int(numberOfDraws / 2), base=13, skip=10
    )
    return np.concatenate((d, 1 - d), axis=1)


mydraws = {
    'HALTON13_ANTI': (
        halton13_anti,
        'Antithetic Halton draws, base 13, skipping 10',
    )
}
database.setRandomNumberGenerators(mydraws)

integrand = exp(bioDraws('U', 'UNIFORM_ANTI'))
simulatedI = MonteCarlo(integrand)

integrand_halton13 = exp(bioDraws('U_halton13', 'HALTON13_ANTI'))
simulatedI_halton13 = MonteCarlo(integrand_halton13)

integrand_mlhs = exp(bioDraws('U_mlhs', 'UNIFORM_MLHS_ANTI'))
simulatedI_mlhs = MonteCarlo(integrand_mlhs)

trueI = exp(1.0) - 1.0

R = 20000

error = simulatedI - trueI

error_halton13 = simulatedI_halton13 - trueI

error_mlhs = simulatedI_mlhs - trueI

simulate = {
    'Analytical Integral': trueI,
    'Simulated Integral': simulatedI,
    'Error             ': error,
    'Simulated Integral (Halton13)': simulatedI_halton13,
    'Error (Halton13)             ': error_halton13,
    'Simulated Integral (MLHS)': simulatedI_mlhs,
    'Error (MLHS)             ': error_mlhs,
}


biogeme = bio.BIOGEME(database, simulate, numberOfDraws=R)
biogeme.modelName = '03antithetic'
results = biogeme.simulate()
print(f"Analytical integral:{results.iloc[0]['Analytical Integral']:.6f}")
print("\t\tUniform (Anti)\t\tHalton13 (Anti)\t\tMLHS (Anti)")
print(
    f"Simulated\t{results.iloc[0]['Simulated Integral']:.6g}\t"
    f"{results.iloc[0]['Simulated Integral (Halton13)']:.6g}\t"
    f"{results.iloc[0]['Simulated Integral (MLHS)']:.6g}"
)
print(
    f"Error\t\t{results.iloc[0]['Error             ']:.6g}\t"
    f"{results.iloc[0]['Error (Halton13)             ']:.6g}\t"
    f"{results.iloc[0]['Error (MLHS)             ']:.6g}"
)
