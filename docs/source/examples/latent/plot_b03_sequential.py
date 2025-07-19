"""

Choice model with latent variables: sequential estimation
=========================================================

Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

Michel Bierlaire, EPFL
Thu May 15 2025, 15:34:13
"""

import sys

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
from biogeme.expressions import Beta, Draws, MonteCarlo, log
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

from read_or_estimate import read_or_estimate
from structural_equations import (
    build_car_centric_attitude,
    build_urban_preference_attitude,
)

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Read the estimates from the structural equation estimation.
MODEL_NAME = 'b01_mimic'
try:
    mimic_results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{MODEL_NAME}.yaml'
    )
except FileNotFoundError:
    print(
        f'Run first the script {MODEL_NAME}.py in order to generate the '
        f'file {MODEL_NAME}.yaml, and move it to the directory saved_results'
    )
    sys.exit()
struct_betas = mimic_results.get_beta_values()

# %%
# Read the estimates from the structural equation estimation.
CHOICE_MODEL_NAME = 'b02_choice_only'
try:
    choice_results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{CHOICE_MODEL_NAME}.yaml'
    )
except FileNotFoundError:
    print(
        f'Run first the script {CHOICE_MODEL_NAME}.py in order to generate the '
        f'file {CHOICE_MODEL_NAME}.yaml, and move it to the directory saved_results'
    )
    sys.exit()
choice_betas = choice_results.get_beta_values()

# %%
# Structural equation: car centric attitude

car_centric_attitude = build_car_centric_attitude(
    estimated_parameters=struct_betas
) + Draws('car_error_term', 'NORMAL_MLHS_ANTI')


# %%
# Latent variable for the urban preference

urban_life_attitude = build_urban_preference_attitude(
    estimated_parameters=struct_betas
) + Draws('urban_error_term', 'NORMAL_MLHS_ANTI')

# %%
# Choice model

# %%
# Parameter from the original choice model
choice_asc_car = Beta('choice_asc_car', choice_betas['choice_asc_car'], None, None, 0)
choice_asc_pt = Beta('choice_asc_pt', choice_betas['choice_asc_pt'], None, None, 0)
choice_beta_cost_hwh = -1.0
choice_beta_cost_other = Beta(
    'choice_beta_cost_other', choice_betas['choice_beta_cost_other'], None, None, 0
)
choice_beta_dist = Beta(
    'choice_beta_dist', choice_betas['choice_beta_dist'], None, None, 0
)
choice_beta_waiting_time = Beta(
    'choice_beta_waiting_time', choice_betas['choice_beta_waiting_time'], None, None, 0
)
choice_beta_time_car = Beta(
    'choice_beta_time_car', choice_betas['choice_beta_time_car'], None, 0, 0
)
choice_beta_time_pt = Beta(
    'choice_beta_time_pt', choice_betas['choice_beta_time_pt'], None, 0, 0
)
scale_choice_model = Beta(
    'scale_choice_model', choice_betas['scale_choice_model'], 1.0e-5, None, 0
)


# %%
# Parameter affected by the latent variables.

# %%
# Alternative specific constants
choice_car_centric_car_cte = Beta('choice_car_centric_car_cte', 0, None, None, 0)
choice_car_centric_pt_cte = Beta('choice_car_centric_pt_cte', 0, None, None, 0)
choice_urban_life_car_cte = Beta('choice_urban_life_car_cte', 0, None, None, 0)
choice_urban_life_pt_cte = Beta('choice_urban_life_pt_cte', 0, None, None, 0)

# %%
# Definition of utility functions:
V0 = scale_choice_model * (
    choice_asc_pt
    + choice_beta_time_pt * TimePT_scaled
    + choice_beta_waiting_time * WaitingTimePT
    + choice_beta_cost_hwh * MarginalCostPT_scaled * PurpHWH
    + choice_beta_cost_other * MarginalCostPT_scaled * PurpOther
    + choice_car_centric_pt_cte * car_centric_attitude
    + choice_urban_life_pt_cte * urban_life_attitude
)

V1 = scale_choice_model * (
    choice_asc_car
    + choice_beta_time_car * TimeCar_scaled
    + choice_beta_cost_hwh * CostCarCHF_scaled * PurpHWH
    + choice_beta_cost_other * CostCarCHF_scaled * PurpOther
    + choice_car_centric_car_cte * car_centric_attitude
    + choice_urban_life_car_cte * urban_life_attitude
)

V2 = scale_choice_model * choice_beta_dist * distance_km_scaled

# %%
# Associate utility functions with the numbering of alternatives
V = {0: V0, 1: V1, 2: V2}

# %%
# Conditional on the latent variables, we have a logit model (called the kernel)
cond_prob = logit(V, None, Choice)

# %%
# We integrate over omega using numerical integration
log_likelihood = log(MonteCarlo(cond_prob))

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, log_likelihood, number_of_draws=10_000)
the_biogeme.model_name = 'b03_sequential'

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
