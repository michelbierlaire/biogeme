"""

Calculation of market shares
============================

We use an estimated model to calculate market shares.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 18:06:27
"""

import sys

from biogeme.biogeme import BIOGEME
from biogeme.data.optima import normalized_weight, read_data
from biogeme.models import nested
from biogeme.results_processing import EstimationResults

from scenarios import scenario

# %%
# Obtain the specification for the default scenario
v, nests, _, _ = scenario()

# %%
# Obtain the expression for the choice probability of each alternative.
prob_pt = nested(v, None, nests, 0)
prob_car = nested(v, None, nests, 1)
prob_sm = nested(v, None, nests, 2)

# %%
# Read the estimation results from the file
try:
    results = EstimationResults.from_yaml_file(
        filename='saved_results/b02estimation.yaml'
    )
except FileNotFoundError:
    sys.exit(
        'Run first the script b02simulation.py '
        'in order to generate the '
        'file b02estimation.yaml.'
    )

# %%
# Read the database
database = read_data()

# %%
# We now simulate the choice probabilities and the weight
simulate = {
    'weight': normalized_weight,
    'Prob. PT': prob_pt,
    'Prob. car': prob_car,
    'Prob. SM': prob_sm,
}

the_biogeme = BIOGEME(database, simulate)
simulated_values = the_biogeme.simulate(results.get_beta_values())

# %%
# We also calculate confidence intervals for the calculated quantities,
b = results.get_betas_for_sensitivity_analysis()
left, right = the_biogeme.confidence_intervals(b, 0.9)

# %%
# Market shares are calculated using the weighted mean of the
# individual probabilities.

# %%
# Alternative car
simulated_values['Weighted prob. car'] = (
    simulated_values['weight'] * simulated_values['Prob. car']
)
left['Weighted prob. car'] = left['weight'] * left['Prob. car']
right['Weighted prob. car'] = right['weight'] * right['Prob. car']

market_share_car = simulated_values['Weighted prob. car'].mean()
market_share_car_left = left['Weighted prob. car'].mean()
market_share_car_right = right['Weighted prob. car'].mean()

# %%
# Alternative public transportation
simulated_values['Weighted prob. PT'] = (
    simulated_values['weight'] * simulated_values['Prob. PT']
)
left['Weighted prob. PT'] = left['weight'] * left['Prob. PT']
right['Weighted prob. PT'] = right['weight'] * right['Prob. PT']

market_share_pt = simulated_values['Weighted prob. PT'].mean()
market_share_pt_left = left['Weighted prob. PT'].mean()
market_share_pt_right = right['Weighted prob. PT'].mean()

# %%
# Alternative slow modes
simulated_values['Weighted prob. SM'] = (
    simulated_values['weight'] * simulated_values['Prob. SM']
)
left['Weighted prob. SM'] = left['weight'] * left['Prob. SM']
right['Weighted prob. SM'] = right['weight'] * right['Prob. SM']

market_share_sm = simulated_values['Weighted prob. SM'].mean()
market_share_sm_left = left['Weighted prob. SM'].mean()
market_share_sm_right = right['Weighted prob. SM'].mean()

# %%
# Reporting.

# %%
# Car.
print(
    f'Market share for car: {100 * market_share_car:.1f}% '
    f'[{100 * market_share_car_left:.1f}%, '
    f'{100 * market_share_car_right:.1f}%]'
)

# %%
# Public transportation.
print(
    f'Market share for PT:  {100 * market_share_pt:.1f}% '
    f'[{100 * market_share_pt_left:.1f}%, '
    f'{100 * market_share_pt_right:.1f}%]'
)

# %%
# Slow modes.
print(
    f'Market share for SM:   {100 * market_share_sm:.1f}% '
    f'[{100 * market_share_sm_left:.1f}%, '
    f'{100 * market_share_sm_right:.1f}%]'
)
