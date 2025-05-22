"""

Choice model with latent variables: simultaneous estimation
===========================================================

Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

Michel Bierlaire, EPFL
Fri May 16 2025, 15:53:52
"""

import sys

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
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
from biogeme.expressions import (
    Beta,
    Draws,
    Elem,
    MonteCarlo,
    MultipleSum,
    Variable,
    exp,
    log,
)
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)
from measurement_equations import all_indicators, generate_measurement_equations
from read_or_estimate import read_or_estimate
from structural_equations import (
    build_car_centric_attitude,
    build_urban_preference_attitude,
)

logger = blog.get_screen_logger(level=blog.DEBUG)

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
# struct_betas = mimic_results.get_beta_values()

# %%
# Read the estimates from the structural equation estimation.
CHOICE_MODEL_NAME = 'b03_sequential'
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
# choice_betas = choice_results.get_beta_values()

# %%
# Structural equation: car centric attitude
sigma_car_structural = Beta('sigma_car_structural', 1, None, None, 0)
car_centric_attitude = build_car_centric_attitude(
    with_beta_parameters=True, estimated_parameters=None
) + sigma_car_structural * Draws('car_error_term', 'NORMAL_MLHS_ANTI')


# %%
# Latent variable for the urban preference

sigma_urban_structural = Beta('sigma_urban_structural', 1, None, None, 0)
urban_preference_attitude = build_urban_preference_attitude(
    with_beta_parameters=True, estimated_parameters=None
) + sigma_urban_structural * Draws('urban_error_term', 'NORMAL_MLHS_ANTI')

# %%
# Choice model

choice_betas = {
    'choice_asc_car': 0.0,
    'choice_asc_sm': 0.0,
    'choice_beta_cost_hwh': 0.0,
    'choice_beta_cost_other': 0.0,
    'choice_beta_dist': 0.0,
    'choice_beta_waiting_time': 0.0,
    'choice_beta_time_car': 0.0,
    'choice_beta_time_pt': 0.0,
}
# %%
# Parameter from the  choice model
choice_asc_car = Beta('choice_asc_car', choice_betas['choice_asc_car'], None, None, 0)
choice_asc_sm = Beta('choice_asc_sm', choice_betas['choice_asc_sm'], None, None, 0)
choice_beta_cost_hwh = Beta(
    'choice_beta_cost_hwh', choice_betas['choice_beta_cost_hwh'], None, None, 0
)
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


# %%
# Parameter affected by the latent variables.

# %%
# Alternative specific constants
choice_car_centric_car_cte = Beta('choice_car_centric_car_cte', 1, None, None, 0)
choice_car_centric_pt_cte = Beta('choice_car_centric_pt_cte', 1, None, None, 0)
choice_urban_life_car_cte = Beta('choice_urban_life_car_cte', 1, None, None, 0)
choice_urban_life_pt_cte = Beta('choice_urban_life_pt_cte', 1, None, None, 0)

# %%
# Definition of utility functions:
V0 = (
    choice_beta_time_pt * TimePT_scaled
    + choice_beta_waiting_time * WaitingTimePT
    + choice_beta_cost_hwh * MarginalCostPT_scaled * PurpHWH
    + choice_beta_cost_other * MarginalCostPT_scaled * PurpOther
    + choice_car_centric_pt_cte * car_centric_attitude
    + choice_urban_life_pt_cte * urban_preference_attitude
)

V1 = (
    choice_asc_car
    + choice_beta_time_car * TimeCar_scaled
    + choice_beta_cost_hwh * CostCarCHF_scaled * PurpHWH
    + choice_beta_cost_other * CostCarCHF_scaled * PurpOther
    + choice_car_centric_car_cte * car_centric_attitude
    + choice_urban_life_car_cte * urban_preference_attitude
)

V2 = choice_asc_sm + choice_beta_dist * distance_km_scaled

# %%
# Associate utility functions with the numbering of alternatives
V = {0: V0, 1: V1, 2: V2}

# %%
# Measurement equations
dict_prob_indicators = generate_measurement_equations(
    car_centric_attitude=car_centric_attitude,
    urban_preference_attitude=urban_preference_attitude,
)

# %%
# We calculate the joint probability of all indicators
log_proba = {
    indicator: log(Elem(dict_prob_indicators[indicator], Variable(indicator)))
    for indicator in all_indicators
}
likelihood_indicator = exp(MultipleSum(log_proba))
# %%
# Conditional on the latent variables, we have a logit model (called the kernel)
cond_prob = logit(V, None, Choice) * likelihood_indicator

# %%
# We integrate over omega using numerical integration
log_likelihood = log(MonteCarlo(cond_prob))

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=10000,
    avoid_analytical_second_derivatives=True,
)
the_biogeme.model_name = 'b03_simultaneous'

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
