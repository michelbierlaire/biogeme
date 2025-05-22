"""

Choice model with latent variables: sequential estimation
=========================================================

Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

Michel Bierlaire, EPFL
Fri May 16 2025, 12:08:11
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
from biogeme.expressions import Beta, Draws, MonteCarlo, exp, log
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)
from read_or_estimate import read_or_estimate
from relevant_data import car_explanatory_variables, urban_explanatory_variables

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Read the estimates from the structural equation estimation.
MODELNAME = 'b01_mimic'
try:
    mimic_results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{MODELNAME}.yaml'
    )
except FileNotFoundError:
    print(
        f'Run first the script {MODELNAME}.py in order to generate the '
        f'file {MODELNAME}.yaml, and move it to the directory saved_results'
    )
    sys.exit()
struct_betas = mimic_results.get_beta_values()

# %%
# Read the estimates from the structural equation estimation.
CHOICE_MODELNAME = 'b03_sequential'
try:
    choice_results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{CHOICE_MODELNAME}.yaml'
    )
except FileNotFoundError:
    print(
        f'Run first the script {CHOICE_MODELNAME}.py in order to generate the '
        f'file {CHOICE_MODELNAME}.yaml, and move it to the directory saved_results'
    )
    sys.exit()
choice_betas = choice_results.get_beta_values()

# %%
# Structural equation: car centric attitude

# %%
# Estimated parameters.
car_struct_coefficients: dict[str, float] = {
    variable_name: struct_betas[f'car_struct_{variable_name}']
    for variable_name in car_explanatory_variables.keys()
}
car_struct_intercept = 0.0

# %%
# Structural equation.
car_centric_attitude = (
    car_struct_intercept
    + sum(
        [
            car_struct_coefficients[variable_name] * variable_expression
            for variable_name, variable_expression in car_explanatory_variables.items()
        ]
    )
    + Draws('car_error_term', 'NORMAL_MLHS_ANTI')
)

# %%
# Latent variable for the urban preference

# %%
# Estimated parameters.
urban_struct_coefficients: dict[str, float] = {
    variable_name: struct_betas[f'urban_struct_{variable_name}']
    for variable_name in urban_explanatory_variables.keys()
}
urban_struct_intercept = 0

# %%
# Structural equation.
urban_life_attitude = (
    urban_struct_intercept
    + sum(
        [
            urban_struct_coefficients[variable_name] * variable_expression
            for variable_name, variable_expression in urban_explanatory_variables.items()
        ]
    )
    + Draws('urban_error_term', 'NORMAL_MLHS_ANTI')
)

# %%
# Choice model

# %%
# Parameter from the original choice model
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
choice_car_centric_car_cte = Beta(
    'choice_car_centric_car_cte',
    choice_betas['choice_car_centric_car_cte'],
    None,
    None,
    0,
)
choice_car_centric_pt_cte = Beta(
    'choice_car_centric_pt_cte',
    choice_betas['choice_car_centric_pt_cte'],
    None,
    None,
    0,
)
choice_urban_life_car_cte = Beta(
    'choice_urban_life_car_cte',
    choice_betas['choice_urban_life_car_cte'],
    None,
    None,
    0,
)
choice_urban_life_pt_cte = Beta(
    'choice_urban_life_pt_cte', choice_betas['choice_urban_life_pt_cte'], None, None, 0
)

choice_beta_waiting_time_ref = Beta(
    'choice_beta_waiting_time_ref',
    choice_betas['choice_beta_waiting_time'],
    None,
    None,
    0,
)
choice_beta_waiting_time_urban = Beta(
    'choice_beta_waiting_time_urban', rm 0, None, None, 0
)
choice_beta_waiting_time = choice_beta_waiting_time_ref * exp(
    choice_beta_waiting_time_urban * urban_life_attitude
)

# %%
# Definition of utility functions:
V0 = (
    choice_beta_time_pt * TimePT_scaled
    + choice_beta_waiting_time * WaitingTimePT
    + choice_beta_cost_hwh * MarginalCostPT_scaled * PurpHWH
    + choice_beta_cost_other * MarginalCostPT_scaled * PurpOther
    + choice_car_centric_pt_cte * car_centric_attitude
    + choice_urban_life_pt_cte * urban_life_attitude
)

V1 = (
    choice_asc_car
    + choice_beta_time_car * TimeCar_scaled
    + choice_beta_cost_hwh * CostCarCHF_scaled * PurpHWH
    + choice_beta_cost_other * CostCarCHF_scaled * PurpOther
    + choice_car_centric_car_cte * car_centric_attitude
    + choice_urban_life_car_cte * urban_life_attitude
)

V2 = choice_asc_sm + choice_beta_dist * distance_km_scaled

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
the_biogeme = BIOGEME(database, log_likelihood, number_of_draws=10000)
the_biogeme.modelName = 'b04_sequential'

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
