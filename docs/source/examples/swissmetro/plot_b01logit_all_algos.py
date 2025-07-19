"""

Logit model
===========

Estimation of a logit model with several algorithms.

Michel Bierlaire, EPFL
Wed Jun 18 2025, 10:03:04
"""

import itertools

import pandas as pd
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.tools import format_timedelta
from tqdm import tqdm

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_sm = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: v_train, 2: v_sm, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = loglogit(V, av, CHOICE)

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
# We run the optimization algorithm with all possible combinations of the parameters.
# The results are stored in a Pandas DataFrame called ``summary``.
results = {}
summary_data = []

# %%
# The first estimation is performed twice, to warm up the python code, so that the execution times are comparable
first = True
for infeasible_cg, initial_radius, second_derivatives in tqdm(
    itertools.product(
        infeasible_cg_values, initial_radius_values, second_derivatives_values
    )
):
    # Create the Biogeme object
    the_biogeme = BIOGEME(
        database,
        logprob,
        infeasible_cg=infeasible_cg,
        initial_radius=initial_radius,
        second_derivatives=second_derivatives,
        generate_html=False,
        generate_yaml=False,
    )

    name = (
        f'cg_{infeasible_cg}_radius_{initial_radius}_second_deriv_{second_derivatives}'
    )
    the_biogeme.model_name = f'b05normal_mixture_algo_{name}'.strip()
    result_data = {
        'InfeasibleCG': infeasible_cg,
        'InitialRadius': initial_radius,
        'SecondDerivatives': second_derivatives,
        'Status': 'Success',  # Assume success unless an exception is caught
    }

    try:
        results[name] = the_biogeme.estimate()
        if first:
            results[name] = the_biogeme.estimate()
            first = False
        opt_time = format_timedelta(
            results[name].optimization_messages["Optimization time"]
        )

        result_data.update(
            {
                'LogLikelihood': results[name].final_log_likelihood,
                'GradientNorm': results[name].gradient_norm,
                'Number of draws': results[name].number_of_draws,
                'Optimization time': opt_time,
                'TerminationCause': results[name].optimization_messages[
                    "Cause of termination"
                ],
            }
        )

    except BiogemeError as e:
        print(e)
        result_data.update(
            {
                'Status': 'Failed',
                'LogLikelihood': None,
                'GradientNorm': None,
                'Number of draws': None,
                'Optimization time': None,
                'TerminationCause': str(e),
            }
        )
        results[name] = None
    summary_data.append(result_data)

summary = pd.DataFrame(summary_data)

# %%
display(summary)

# %%
SUMMARY_FILE = '01logit_all_algos.csv'
summary.to_csv(SUMMARY_FILE, index=False)
print(f'Summary reported in file {SUMMARY_FILE}')
