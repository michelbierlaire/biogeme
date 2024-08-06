"""

Mixture of logit
================

Example of the use of different algorithms to estimate the model.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:38:34 2023

"""

import itertools
import pandas as pd
from biogeme.tools import format_timedelta
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
import biogeme.exceptions as excep
from biogeme.expressions import Beta, bioDraws, log, MonteCarlo

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b05normal_mixture_all_algos.py')

# %%
# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
B_TIME = Beta('B_TIME', 0, None, None, 0)

# %%
# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional to B_TIME_RND, we have a logit model (called the kernel)
prob = models.logit(V, av, CHOICE)

# %%
# We integrate over B_TIME_RND using Monte-Carlo
logprob = log(MonteCarlo(prob))

# %%
# Options for the optimization algorithm
# --------------------------------------

# %%
# The conjugate gradient iteration can be constrained to stay feasible, or not.
infeasible_cg_values = [True, False]

# %%
# The radius of the first trust region is tested with three different values.
initial_radius_values = [0.1, 1.0, 10.0]

# %%
# The percentage of iterations such that the analytical second derivatives is evaluated.
second_derivatives_values = [0.0, 0.5, 1.0]

# %%
# We run the optimization algorithm with all possible combinations of the parameters. The results are stored in a Pandas DataFrame called ``summary``.
results = {}
summary = pd.DataFrame(
    columns=[
        'LogLikelihood',
        'GradientNorm',
        'Optimization time',
        'TerminationCause',
        'Status',
    ]
)

for infeasible_cg, initial_radius, second_derivatives in itertools.product(
    infeasible_cg_values, initial_radius_values, second_derivatives_values
):
    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database, logprob, parameter_file='few_draws.toml')
    # We set the parameters of the optimization algorithm
    the_biogeme.infeasible_cg = infeasible_cg
    the_biogeme.initial_radius = initial_radius
    the_biogeme.second_derivatives = second_derivatives
    # We cancel the generation of the outputfiles
    the_biogeme.generate_html = False
    the_biogeme.generate_pickle = False

    name = (
        f'cg_{infeasible_cg}_radius_{initial_radius}_second_deriv_{second_derivatives}'
    )
    the_biogeme.modelName = f'b05normal_mixture_algo_{name}'.strip()
    result_data = {
        'InfeasibleCG': infeasible_cg,
        'InitialRadius': initial_radius,
        'SecondDerivatives': second_derivatives,
        'Status': 'Success',  # Assume success unless an exception is caught
    }

    try:
        results[name] = the_biogeme.estimate()
        opt_time = format_timedelta(
            results[name].data.optimizationMessages["Optimization time"]
        )

        result_data.update(
            {
                'LogLikelihood': results[name].data.logLike,
                'GradientNorm': results[name].data.gradientNorm,
                'Optimization time': opt_time,
                'TerminationCause': results[name].data.optimizationMessages[
                    "Cause of termination"
                ],
            }
        )

    except excep.BiogemeError as e:
        print(e)
        result_data.update(
            {
                'Status': 'Failed',
                'LogLikelihood': None,
                'GradientNorm': None,
                'Optimization time': None,
                'TerminationCause': str(e),
            }
        )
        results[name] = None

    summary = pd.concat([summary, pd.DataFrame([result_data])], ignore_index=True)

# %%
summary

# %%
SUMMARY_FILE = '05normalMixture_allAlgos.csv'
summary.to_csv(SUMMARY_FILE, index=False)
print(f'Summary reported in file {SUMMARY_FILE}')
