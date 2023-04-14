"""File: b03antithetic_explicit.py

 Author: Michel Bierlaire, EPFL
 Date: Thu Apr 13 20:49:50 2023

Calculation of a simple integral using Monte-Carlo integration. It
illustrates how to use antothetic draws, explicitly generared.
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import draws
from biogeme.expressions import exp, bioDraws, MonteCarlo

R = 10000

# We create a fake database with one entry, as it is required to store
# the draws
pandas = pd.DataFrame()
pandas['FakeColumn'] = [1.0]
database = db.Database('fake_database', pandas)


def halton13(sample_size, number_of_draws):
    """The user can define new draws. For example, Halton draws with base
    13, skipping the first 10 draws.

    """
    return draws.getHaltonDraws(sample_size, number_of_draws, base=13, skip=10)


mydraws = {'HALTON13': (halton13, 'Halton draws, base 13, skipping 10')}
database.setRandomNumberGenerators(mydraws)

U = bioDraws('U', 'UNIFORM')
integrand = exp(U) + exp(1 - U)
simulatedI = MonteCarlo(integrand) / 2.0

U_halton13 = bioDraws('U_halton13', 'HALTON13')
integrand_halton13 = exp(U_halton13) + exp(1 - U_halton13)
simulatedI_halton13 = MonteCarlo(integrand_halton13) / 2.0

U_mlhs = bioDraws('U_mlhs', 'UNIFORM_MLHS')
integrand_mlhs = exp(U_mlhs) + exp(1 - U_mlhs)
simulatedI_mlhs = MonteCarlo(integrand_mlhs) / 2.0

trueI = exp(1.0) - 1.0

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

biosim = bio.BIOGEME(database, simulate)
biosim.modelName = 'b03antithetic_explicit'
results = biosim.simulate(theBetaValues={})
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
