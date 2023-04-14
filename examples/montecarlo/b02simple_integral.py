"""File: b02simple_integral.py

 Author: Michel Bierlaire, EPFL
 Date: Thu Apr 13 20:46:01 2023

Calculation of a simple integral using numerical integration and
Monte-Carlo integration with various types of draws, including Halton
draws base 13. It illustrates how to use draws that are not directly
available in Biogeme.

"""


import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import draws
from biogeme.expressions import exp, bioDraws, MonteCarlo


# We create a fake database with one entry, as it is required
# to store the draws
pandas = pd.DataFrame()
pandas['FakeColumn'] = [1.0]
database = db.Database('fakeDatabase', pandas)


def halton13(sample_size, number_of_draws):
    """
    The user can define new draws. For example, Halton draws
    with base 13, skipping the first 10 draws.
    """
    return draws.getHaltonDraws(sample_size, number_of_draws, base=13, skip=10)


mydraws = {'HALTON13': (halton13, 'Halton draws, base 13, skipping 10')}
database.setRandomNumberGenerators(mydraws)

integrand = exp(bioDraws('U', 'UNIFORM'))
simulatedI = MonteCarlo(integrand)

integrand_halton = exp(bioDraws('U_halton', 'UNIFORM_HALTON2'))
simulatedI_halton = MonteCarlo(integrand_halton)

integrand_halton13 = exp(bioDraws('U_halton13', 'HALTON13'))
simulatedI_halton13 = MonteCarlo(integrand_halton13)

integrand_mlhs = exp(bioDraws('U_mlhs', 'UNIFORM_MLHS'))
simulatedI_mlhs = MonteCarlo(integrand_mlhs)

trueI = exp(1.0) - 1.0

R = 20000

sampleVariance = MonteCarlo(integrand * integrand) - simulatedI * simulatedI
stderr = (sampleVariance / R) ** 0.5
error = simulatedI - trueI

sampleVariance_halton = (
    MonteCarlo(integrand_halton * integrand_halton)
    - simulatedI_halton * simulatedI_halton
)
stderr_halton = (sampleVariance_halton / R) ** 0.5
error_halton = simulatedI_halton - trueI

sampleVariance_halton13 = (
    MonteCarlo(integrand_halton13 * integrand_halton13)
    - simulatedI_halton13 * simulatedI_halton13
)
stderr_halton13 = (sampleVariance_halton13 / R) ** 0.5
error_halton13 = simulatedI_halton13 - trueI

sampleVariance_mlhs = (
    MonteCarlo(integrand_mlhs * integrand_mlhs) - simulatedI_mlhs * simulatedI_mlhs
)
stderr_mlhs = (sampleVariance_mlhs / R) ** 0.5
error_mlhs = simulatedI_mlhs - trueI


simulate = {
    'Analytical Integral': trueI,
    'Simulated Integral': simulatedI,
    'Sample variance   ': sampleVariance,
    'Std Error         ': stderr,
    'Error             ': error,
    'Simulated Integral (Halton)': simulatedI_halton,
    'Sample variance (Halton)   ': sampleVariance_halton,
    'Std Error (Halton)         ': stderr_halton,
    'Error (Halton)             ': error_halton,
    'Simulated Integral (Halton13)': simulatedI_halton13,
    'Sample variance (Halton13)   ': sampleVariance_halton13,
    'Std Error (Halton13)         ': stderr_halton13,
    'Error (Halton13)             ': error_halton13,
    'Simulated Integral (MLHS)': simulatedI_mlhs,
    'Sample variance (MLHS)   ': sampleVariance_mlhs,
    'Std Error (MLHS)         ': stderr_mlhs,
    'Error (MLHS)             ': error_mlhs,
}

biosim = bio.BIOGEME(database, simulate)
biosim.modelName = 'b02simple_integral'
results = biosim.simulate(theBetaValues={})
print(f'Analytical integral:{results.iloc[0]["Analytical Integral"]:.6g}')
print('\t\tUniform\t\tHalton\t\tHalton13\tMLHS')
print(
    f'Simulated\t{results.iloc[0]["Simulated Integral"]:.6g}\t'
    f'{results.iloc[0]["Simulated Integral (Halton)"]:.6g}\t'
    f'{results.iloc[0]["Simulated Integral (Halton13)"]:.6g}\t'
    f'{results.iloc[0]["Simulated Integral (MLHS)"]:.6g}'
)
print(
    f'Sample var.\t{results.iloc[0]["Sample variance   "]:.6g}\t'
    f'{results.iloc[0]["Sample variance (Halton)   "]:.6g}\t'
    f'{results.iloc[0]["Sample variance (Halton13)   "]:.6g}\t'
    f'{results.iloc[0]["Sample variance (MLHS)   "]:.6g}'
)
print(
    f'Std error\t{results.iloc[0]["Std Error         "]:.6g}\t'
    f'{results.iloc[0]["Std Error (Halton)         "]:.6g}\t'
    f'{results.iloc[0]["Std Error (Halton13)         "]:.6g}\t'
    f'{results.iloc[0]["Std Error (MLHS)         "]:.6g}'
)
print(
    f'Error\t\t{results.iloc[0]["Error             "]:.6g}\t'
    f'{results.iloc[0]["Error (Halton)             "]:.6g}\t'
    f'{results.iloc[0]["Error (Halton13)             "]:.6g}\t'
    f'{results.iloc[0]["Error (MLHS)             "]:.6g}'
)
