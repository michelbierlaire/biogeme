"""

Calculation of revenues
=======================

We use an estimated model to calculate revenues.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 21:02:19 2023

"""

import sys
import numpy as np

try:
    import matplotlib.pyplot as plt

    can_plot = True
except ModuleNotFoundError:
    can_plot = False
from tqdm import tqdm
from biogeme import models
from biogeme.exceptions import BiogemeError
import biogeme.biogeme as bio
import biogeme.results as res
from biogeme.data.optima import read_data, normalized_weight
from scenarios import scenario

# %%
# Read the estimation results from the file.
try:
    results = res.bioResults(pickle_file='saved_results/b02estimation.pickle')
except BiogemeError:
    sys.exit(
        'Run first the script b02simulation.py '
        'in order to generate the '
        'file b02estimation.pickle.'
    )

# %%
# Read the data
database = read_data()


# %%
# Function calculating the revenues
def revenues(factor: float) -> tuple[float, float, float]:
    """Calculate the total revenues generated by public transportation,
        when the price is multiplied by a factor.

    :param factor: factor that multiplies the current cost of public
        transportation

    :return: total revenues, followed by the lower and upper bound of
        the confidence interval.

    """
    # Obtain the specification for the default scenario
    V, nests, _, marginal_cost_scenario = scenario(factor=factor)

    # Obtain the expression for the choice probability of each alternative
    prob_pt = models.nested(V, None, nests, 0)

    # We now simulate the choice probabilities,the weight and the
    # price variable

    simulate = {
        'weight': normalized_weight,
        'Revenue public transportation': prob_pt * marginal_cost_scenario,
    }

    the_biogeme = bio.BIOGEME(database, simulate)
    simulated_values = the_biogeme.simulate(results.get_beta_values())

    # We also calculate confidence intervals for the calculated quantities

    betas = the_biogeme.free_beta_names
    beta_bootstrap = results.get_betas_for_sensitivity_analysis(betas)
    left, right = the_biogeme.confidence_intervals(beta_bootstrap, 0.9)

    revenues_pt = (
        simulated_values['Revenue public transportation'] * simulated_values['weight']
    ).sum()
    revenues_pt_left = (left['Revenue public transportation'] * left['weight']).sum()
    revenues_pt_right = (right['Revenue public transportation'] * right['weight']).sum()
    return revenues_pt, revenues_pt_left, revenues_pt_right


# %%
# Current revenues for public transportation
r, r_left, r_right = revenues(factor=1.0)
print(
    f'Total revenues for public transportation (for the sample): {r:.1f} CHF '
    f'[{r_left:.1f} CHF, '
    f'{r_right:.1f} CHF]'
)

# %%
# We now investigate how the revenues vary with the multiplicative factor

factors = np.arange(0.0, 5.0, 0.05)
plot_revenues = [revenues(s) for s in tqdm(factors)]
zipped = zip(*plot_revenues)
rev = next(zipped)
lower = next(zipped)
upper = next(zipped)

largest_revenue = max(rev)
max_index = rev.index(largest_revenue)

# %%
print(
    f'Largest revenue: {largest_revenue:.1f} obtained with '
    f'factor {factors[max_index]:.1f}'
)

# %%
if can_plot:
    # We plot the results
    ax = plt.gca()
    ax.plot(factors, rev, label="Revenues")
    ax.plot(factors, lower, label="Lower bound of the CI")
    ax.plot(factors, upper, label="Upper bound of the CI")
    ax.legend()

    plt.show()