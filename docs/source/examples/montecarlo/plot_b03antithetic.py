"""

Antithetic draws
================

Calculation of a simple integral using Monte-Carlo integration. It
illustrates how to use antithetic draws.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 21:08:23
"""

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.draws import RandomNumberGeneratorTuple, get_halton_draws
from biogeme.expressions import Draws, MonteCarlo, exp

# %%
# We create a fake database with one entry, as it is required to store
# the draws.
df = pd.DataFrame()
df['FakeColumn'] = [1.0]
database = Database('fake_database', df)


# %%
def halton13_anti(sample_size: int, number_of_draws: int) -> np.array:
    """The user can define new draws. For example, antithetic Halton
    draws with base 13, skipping the first 10 draws.

    :param sample_size: number of observations for which draws must be
                       generated.
    :param number_of_draws: number of draws to generate.

    """

    # We first generate half of the number of requested draws.
    the_draws = get_halton_draws(
        sample_size, int(number_of_draws / 2.0), base=13, skip=10
    )
    # We complement them with their antithetic version.
    return np.concatenate((the_draws, 1 - the_draws), axis=1)


# %%
my_draws = {
    'HALTON13_ANTI': RandomNumberGeneratorTuple(
        halton13_anti,
        'Antithetic Halton draws, base 13, skipping 10',
    )
}

# %%
# Integrate with antithetic pseudo-random number generator.
integrand = exp(Draws('U', 'UNIFORM_ANTI'))
simulated_integral = MonteCarlo(integrand)

# %%
# Integrate with antithetic Halton draws, base 13.
integrand_halton13 = exp(Draws('U_halton13', 'HALTON13_ANTI'))
simulated_integral_halton13 = MonteCarlo(integrand_halton13)

# %%
# Integrate with antithetic MLHS.
integrand_mlhs = exp(Draws('U_mlhs', 'UNIFORM_MLHS_ANTI'))
simulated_integral_mlhs = MonteCarlo(integrand_mlhs)

# %%
# True value
true_integral = exp(1.0) - 1.0

# %%
# Number of draws.
R = 2_000_000

# %%
error = simulated_integral - true_integral

# %%
error_halton13 = simulated_integral_halton13 - true_integral

# %%
error_mlhs = simulated_integral_mlhs - true_integral

# %%
simulate = {
    'Analytical Integral': true_integral,
    'Simulated Integral': simulated_integral,
    'Error             ': error,
    'Simulated Integral (Halton13)': simulated_integral_halton13,
    'Error (Halton13)             ': error_halton13,
    'Simulated Integral (MLHS)': simulated_integral_mlhs,
    'Error (MLHS)             ': error_mlhs,
}


# %%
biosim = BIOGEME(
    database, simulate, random_number_generators=my_draws, number_of_draws=R
)
biosim.modelName = 'b03antithetic'
results = biosim.simulate(the_beta_values={})
display(results)

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
