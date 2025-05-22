"""

Simple integral
===============

Calculation of a simple integral using Monte-Carlo integration.

Michel Bierlaire, EPFL
Tue Apr 29 2025, 11:43:13
"""

import pandas as pd

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Draws, MonteCarlo, exp

# %%
# We create a fake database with one entry, as it is required
# to store the draws
df = pd.DataFrame()
df['FakeColumn'] = [1.0]
database = Database('fake_database', df)

# %%
integrand = exp(Draws('U', 'UNIFORM'))
simulated_integral = MonteCarlo(integrand)

# %%
true_integral = exp(1.0) - 1.0

# %%
R = 200
MULTIPLIER = 100000


sample_variance = (
    MonteCarlo(integrand * integrand) - simulated_integral * simulated_integral
)
stderr = (sample_variance / R) ** 0.5
error = simulated_integral - true_integral

simulate = {
    'Analytical Integral': true_integral,
    'Simulated Integral': simulated_integral,
    'Sample variance   ': sample_variance,
    'Std Error         ': stderr,
    'Error             ': error,
}

biosim = BIOGEME(database, simulate, number_of_draws=R)
R = biosim.number_of_draws
biosim.model_name = f'01simpleIntegral_{R}'
results = biosim.simulate(the_beta_values={})
print(f'Number of draws: {R}')
for c in results.columns:
    print(f'{c}: {results.loc[0, c]}')

# %%
# Change the number of draws
biogeme2 = BIOGEME(database, simulate, number_of_draws=R * MULTIPLIER)
biogeme2.model_name = '01simpleIntegral_{multiplier*R}'
results2 = biogeme2.simulate(the_beta_values={})
print(f'Number of draws: {MULTIPLIER * R}')
for c in results2.columns:
    print(f'{c}: {results2.loc[0, c]}')
