"""File 05revenues.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 11:26:34 2021

We use an estimated model to calculate revenues
"""
import sys
import numpy as np
from tqdm import tqdm
from biogeme import models
import biogeme.exceptions as excep
import matplotlib.pyplot as plt
import biogeme.biogeme as bio
import biogeme.results as res
from scenarios import scenario, database, normalizedWeight

# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='02estimation.pickle')
except excep.biogemeError:
    sys.exit(
        'Run first the script 02simulation.py '
        'in order to generate the '
        'file 02estimation.pickle.'
    )


def revenues(factor):
    """Calculate the total revenues generated by public transportation,
        when the price is multiplied by a factor.

    :param factor: factor that multiplies the current cost of public
        transportation
    :type factor: float

    :return: total revenues, followed by the lower and upper bound of
        the confidence interval.
    :rtype: tuple(float, float, float)

    """
    # Obtain the specification for the default scenario
    V, nests, _, MarginalCostScenario = scenario(factor=factor)

    # Obtain the expression for the choice probability of each alternative
    prob_PT = models.nested(V, None, nests, 0)

    # We now simulate the choice probabilities,the weight and the
    # price variable

    simulate = {
        'weight': normalizedWeight,
        'Revenue public transportation': prob_PT * MarginalCostScenario,
    }

    biogeme = bio.BIOGEME(database, simulate)
    simulated_values = biogeme.simulate(results.getBetaValues())

    # We also calculate confidence intervals for the calculated quantities

    betas = biogeme.freeBetaNames()
    b = results.getBetasForSensitivityAnalysis(betas)
    left, right = biogeme.confidenceIntervals(b, 0.9)

    revenues_pt = (
        simulated_values['Revenue public transportation']
        * simulated_values['weight']
    ).sum()
    revenues_pt_left = (
        left['Revenue public transportation'] * left['weight']
    ).sum()
    revenues_pt_right = (
        right['Revenue public transportation'] * right['weight']
    ).sum()
    return revenues_pt, revenues_pt_left, revenues_pt_right


# Current revenues for public transportation

r, r_left, r_right = revenues(factor=1.0)
print(
    f'Total revenues for public transportation (for the sample): {r:.1f} CHF '
    f'[{r_left:.1f} CHF, '
    f'{r_right:.1f} CHF]'
)

# We now investigate how the revenues vary with the multiplicative factor

factors = np.arange(0.0, 5.0, 0.05)

plot_revenues = [revenues(s) for s in tqdm(factors)]
zipped = zip(*plot_revenues)
rev = next(zipped)
lower = next(zipped)
upper = next(zipped)

largest_revenue = max(rev)
max_index = rev.index(largest_revenue)
print(
    f'Largest revenue: {largest_revenue:.1f} obtained with '
    f'factor {factors[max_index]:.1f}'
)

# We plot the results
ax = plt.gca()
ax.plot(factors, rev, label="Revenues")
ax.plot(factors, lower, label="Lower bound of the CI")
ax.plot(factors, upper, label="Upper bound of the CI")
ax.legend()

plt.show()
