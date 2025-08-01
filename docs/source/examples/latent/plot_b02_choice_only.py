"""

Estimation of choice model without latent variables
===================================================

Michel Bierlaire, EPFL
Thu May 15 2025, 15:23:42
"""

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.data.optima import (
    Choice,
    CostCarCHF_scaled,
    MarginalCostPT_scaled,
    PurpHWH,
    PurpOther,
    TimeCar_scaled,
    TimePT_scaled,
    WaitingTimePT,
    distance_km_scaled,
    read_data,
)
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import (
    get_pandas_estimated_parameters,
)

from read_or_estimate import read_or_estimate

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Choice model: parameters
choice_asc_car = Beta('choice_asc_car', 0.0, None, None, 0)
choice_asc_pt = Beta('choice_asc_pt', 0, None, None, 0)
choice_asc_sm = Beta('choice_asc_sm', 0, None, None, 1)
choice_beta_cost_hwh = Beta('choice_beta_cost_hwh', -1, None, None, 1)
choice_beta_cost_other = Beta('choice_beta_cost_other', 0, None, None, 0)
choice_beta_dist = Beta('choice_beta_dist', 0, None, None, 0)
choice_beta_time_car = Beta('choice_beta_time_car', 0, None, 0, 0)
choice_beta_time_pt = Beta('choice_beta_time_pt', 0, None, 0, 0)
choice_beta_waiting_time = Beta('choice_beta_waiting_time', 0, None, None, 0)
scale_choice_model = Beta('scale_choice_model', 1, 1.0e-5, 10, 0)

# %%
# Definition of utility functions:
V0 = scale_choice_model * (
    choice_asc_pt
    + choice_beta_time_pt * TimePT_scaled
    + choice_beta_waiting_time * WaitingTimePT
    + choice_beta_cost_hwh * MarginalCostPT_scaled * PurpHWH
    + choice_beta_cost_other * MarginalCostPT_scaled * PurpOther
)

V1 = scale_choice_model * (
    choice_asc_car
    + choice_beta_time_car * TimeCar_scaled
    + choice_beta_cost_hwh * CostCarCHF_scaled * PurpHWH
    + choice_beta_cost_other * CostCarCHF_scaled * PurpOther
)

V2 = scale_choice_model * (choice_asc_sm + choice_beta_dist * distance_km_scaled)

# %%
# Associate utility functions with the numbering of alternatives
V = {0: V0, 1: V1, 2: V2}

# %%
# We integrate over omega using numerical integration
log_likelihood = loglogit(V, None, Choice)

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, log_likelihood)
the_biogeme.model_name = 'b02_choice_only'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results = read_or_estimate(the_biogeme=the_biogeme, directory='saved_results')

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
