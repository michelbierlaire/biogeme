"""

1c. Illustration of the quick_estimate method in Biogeme
=======================================================

This example estimates the same logit model as in Example 1a, but uses
`quick_estimate`, a lightweight estimation procedure designed for cases
where only the estimated parameter values are required.

Unlike the standard estimation procedure, `quick_estimate` skips the
calculation of second-order derivatives and several post-estimation
statistics. As a result, the estimation is faster, but some indicators
normally reported by Biogeme are not available.

The script illustrates:

- the specification of a logit model,
- the use of `quick_estimate`,
- the inspection of the estimation results,
- the extraction of the estimated parameters into a pandas DataFrame,
- the manual generation of a YAML output file.

The `# %%` markers are used to separate the script into notebook cells
when the example gallery is converted into Jupyter notebooks.

Tested with Biogeme 3.3.3.

Michel Bierlaire, EPFL
Tue Jun 09 2026, 14:45:00
"""

from IPython.core.display_functions import display

# %%
# Import the variables and the database prepared in the Swissmetro data-processing example.
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

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01logit_ter.py')

# %%
# Define the model parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)


# %%
# Define the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate each utility function with the corresponding alternative identifier.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with each alternative.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Define the log-likelihood contribution of each observation.
logprob = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, logprob)
the_biogeme.model_name = 'b01c_logit'

# %%
# Calculate the null log likelihood used in the estimation report.
the_biogeme.calculate_null_loglikelihood(av)

# %%
# Estimate the parameters using the quick estimation procedure.
results = the_biogeme.quick_estimate()

# %%
# Display a short summary of the estimation results.
print(results.short_summary())

# %%
# The quick estimation procedure does not calculate the initial log likelihood,
# second derivatives, or several post-estimation statistics. It is useful
# when only the estimated parameter values are required.

# %%
# Convert the estimated parameters into a pandas DataFrame.
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)

# %%
# Display the available general estimation statistics.
print('General statistics')
print('------------------')
stats = results.get_general_statistics()
for description, value in stats.items():
    print(f'{description}: {value}')

# %%
# The YAML file is not generated automatically when quick_estimate is used.
# Generate it manually if needed.
results.dump_yaml_file(filename=f'{the_biogeme.model_name}.yaml')
