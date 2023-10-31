"""File b13panel_simul.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 18:16:29 2023

 Example of a mixture of logit models, using Monte-Carlo integration.
 The datafile is organized as panel data.
 Three alternatives: Train, Car and Swissmetro

"""

import sys
import pickle
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
import biogeme.exceptions as excep
import biogeme.results as res
from biogeme.expressions import (
    Beta,
    bioDraws,
    PanelLikelihoodTrajectory,
    MonteCarlo,
    log,
)
from swissmetro_panel import (
    database,
    CHOICE,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
)

NUMBER_OF_DRAWS = 1000

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b13panel_simul.py')

# Parameters to be estimated
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation
B_TIME = Beta('B_TIME', 0, None, None, 0)

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL_ANTI')

# We do the same for the constants, to address serial correlation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR_S = Beta('ASC_CAR_S', 1, None, None, 0)
ASC_CAR_RND = ASC_CAR + ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL_ANTI')

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_TRAIN_S = Beta('ASC_TRAIN_S', 1, None, None, 0)
ASC_TRAIN_RND = ASC_TRAIN + ASC_TRAIN_S * bioDraws('ASC_TRAIN_RND', 'NORMAL_ANTI')

ASC_SM = Beta('ASC_SM', 0, None, None, 1)
ASC_SM_S = Beta('ASC_SM_S', 1, None, None, 0)
ASC_SM_RND = ASC_SM + ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL_ANTI')

# Definition of the utility functions
V1 = ASC_TRAIN_RND + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM_RND + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR_RND + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Conditional to the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel)
obsprob = models.logit(V, av, CHOICE)

# Conditional to the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
condprobIndiv = PanelLikelihoodTrajectory(obsprob)

# We integrate over the random parameters using Monte-Carlo
logprob = log(MonteCarlo(condprobIndiv))

# Estimate the parameters.
try:
    results = res.bioResults(pickleFile='b12panel.pickle')
except excep.BiogemeError:
    sys.exit(
        'Run first the script b12panel.py '
        'in order to generate the '
        'file b12panel.pickle.'
    )

# Simulate to recalculate the log likelihood directly from the
# formula, without the Biogeme object
simulated_loglike = logprob.getValue_c(
    database=database,
    betas=results.getBetaValues(),
    numberOfDraws=NUMBER_OF_DRAWS,
    aggregation=True,
    prepareIds=True,
)

print(f'Simulated log likelihood: {simulated_loglike}')

numerator = MonteCarlo(B_TIME_RND * condprobIndiv)
denominator = MonteCarlo(condprobIndiv)
simulate = {
    'Numerator': numerator,
    'Denominator': denominator,
}

biosim = bio.BIOGEME(database, simulate)
sim = biosim.simulate(results.getBetaValues())

sim['Individual-level parameters'] = sim['Numerator'] / sim['Denominator']
PICKLE_FILE = 'b13panel_individual_parameters.pickle'
with open(PICKLE_FILE, 'wb') as f:
    pickle.dump(sim, f)

HTML_FILE = 'b13panel_simul.html'
with open(HTML_FILE, 'w', encoding='utf-8') as h:
    print(sim.to_html(), file=h)
print(f'Simulated values available in {HTML_FILE}')
