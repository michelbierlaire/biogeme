"""

1a. Estimation of a logit model (Bayesian)
==========================================

 Three alternatives:

   - train,
   - car and,
   - Swissmetro.

 Stated preferences data.

Michel Bierlaire, EPFL
Thu Oct 30 2025, 10:15:52
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import (
    BayesianResults,
    FigureSize,
    generate_html_file as generate_bayesian_html_file,
    get_pandas_estimated_parameters,
)
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.filenames import get_new_file_name
from biogeme.models import loglogit

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
# The logger sets the verbosity of Biogeme. By default, Biogeme is quite silent and generates only warnings.
# To have more information about what it happening behind the scene, the level should be set to `blog.INFO`.
logger = blog.get_screen_logger(level=blog.DEBUG)
logger.info('Example b01a_logit.py')


# %%
# Parameters to be estimated: alternative specific constants.
# By default, the prior distribution is normal, possibly truncated if bounds are defined, with the mean
# defined by the user, and scale parameter 10.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)

# %%
# The constant associated with Swissmetro is normalized to zero. It does not need to be defined at all.
# Here, we illustrate the fact that setting the last argument of the `Beta` function to 1 fixes the parameter
# to its default value (here, 0).
asc_sm = Beta('asc_sm', 0, None, None, 0)

# %%
# Coefficients of the attributes. It is useful to set the upper bound to 0 to reflect the prior assumption about
# the sign of those parameters.
b_time = Beta('b_time', 0, None, 0, 0)
b_cost = Beta('b_cost', 0, None, 0, 0)


# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_sm = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_sm, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'overspecification'

# %%
# Estimate the parameters.
try:
    results = BayesianResults.from_netcdf(filename=f'{the_biogeme.model_name}.nc')
    html_filename = get_new_file_name(the_biogeme.model_name, 'html')
    generate_bayesian_html_file(
        filename=html_filename,
        estimation_results=results,
        figure_size=FigureSize.LARGE,
    )
except FileNotFoundError:
    results: BayesianResults = the_biogeme.bayesian_estimation()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)

# %%
# Describe the draws stored in the PyMC report.
display(results.report_stored_variables())
