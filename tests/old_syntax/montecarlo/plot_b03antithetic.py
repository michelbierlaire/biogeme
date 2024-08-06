"""

Antithetic draws
================

Calculation of a simple integral using Monte-Carlo integration. It
illustrates how to use antithetic draws.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 20:48:02 2023
"""

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import draws
from biogeme.expressions import exp, bioDraws, MonteCarlo

# %%
# We create a fake database with one entry, as it is required to store
# the draws.
df = pd.DataFrame()
df['FakeColumn'] = [1.0]
database = db.Database('fake_database', df)


# %%
def halton13_anti(sample_size: int, number_of_draws: int) -> np.array:
    """The user can define new draws. For example, antithetic Halton
    draws with base 13, skipping the first 10 draws.

    :param sample_size: number of observations for which draws must be
                       generated.
    :param number_of_draws: number of draws to generate.

    """

    # We first generate half of the number of requested draws.
    the_draws = draws.getHaltonDraws(
        sample_size, int(number_of_draws / 2.0), base=13, skip=10
    )
    # We complement them with their antithetic version.
    return np.concatenate((the_draws, 1 - the_draws), axis=1)


# %%
mydraws = {
    'HALTON13_ANTI': (
        halton13_anti,
        'Antithetic Halton draws, base 13, skipping 10',
    )
}
database.setRandomNumberGenerators(mydraws)

# %%
# Integrate with antithetic pseudo-random number generator.
integrand = exp(bioDraws('U', 'UNIFORM_ANTI'))
simulatedI = MonteCarlo(integrand)

# %%
# Integrate with antithetic Halton draws, base 13.
integrand_halton13 = exp(bioDraws('U_halton13', 'HALTON13_ANTI'))
simulatedI_halton13 = MonteCarlo(integrand_halton13)

# %%
# Integrate with antithetic MLHS.
integrand_mlhs = exp(bioDraws('U_mlhs', 'UNIFORM_MLHS_ANTI'))
simulatedI_mlhs = MonteCarlo(integrand_mlhs)

# %%
# True value
trueI = exp(1.0) - 1.0

# %%
# Number of draws.
R = 20000

# %%
error = simulatedI - trueI

# %%
error_halton13 = simulatedI_halton13 - trueI

# %%
error_mlhs = simulatedI_mlhs - trueI

# %%
simulate = {
    'Analytical Integral': trueI,
    'Simulated Integral': simulatedI,
    'Error             ': error,
    'Simulated Integral (Halton13)': simulatedI_halton13,
    'Error (Halton13)             ': error_halton13,
    'Simulated Integral (MLHS)': simulatedI_mlhs,
    'Error (MLHS)             ': error_mlhs,
}


# %%
biosim = bio.BIOGEME(database, simulate)
biosim.modelName = 'b03antithetic'
results = biosim.simulate(theBetaValues={})
results

# %%
# Reorganize the results.
print(f"Analytical integral: {results.iloc[0]['Analytical Integral']:.6f}")
print(
    f"         \t{'Uniform (Anti)':>15}\t{'Halton13 (Anti)':>15}\t{'MLHS (Anti)':>15}"
)
print(
    f"Simulated\t{results.iloc[0]['Simulated Integral']:>15.6g}\t"
    f"{results.iloc[0]['Simulated Integral (Halton13)']:>15.6g}\t"
    f"{results.iloc[0]['Simulated Integral (MLHS)']:>15.6g}"
)
print(
    f"Error\t\t{results.iloc[0]['Error             ']:>15.6g}\t"
    f"{results.iloc[0]['Error (Halton13)             ']:>15.6g}\t"
    f"{results.iloc[0]['Error (MLHS)             ']:>15.6g}"
)
