"""
Baseline mode choice model: maximum likelihood estimation
=========================================================

This example estimates a multinomial logit model for transportation mode
choice using maximum likelihood and Biogeme.

It is the baseline specification of the hybrid-choice tutorial. The model
contains only observed choice variables and standard utility functions: there
are no latent variables, structural equations, or measurement equations. The
results therefore provide a reference against which the hybrid-choice
specifications introduced later can be compared.

The script performs the following steps:

- load the Optima mode-choice data,
- build the utility functions from the common tutorial specification,
- construct the logit log-likelihood,
- estimate the model, or reload previously saved estimation results,
- display the estimated parameters as pandas and LaTeX tables.

Michel Bierlaire
Sat Jun 06 2026, 15:23:41
"""

from choice_latent_variables import generate_utility_functions
from optima import (
    Choice,
    read_data,
)

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.models import loglogit
from biogeme.results_processing import (
    get_latex_estimated_parameters,
    get_latex_general_statistics,
    get_pandas_estimated_parameters,
)

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Load the Optima mode-choice data.
database = read_data()

# Build the utility functions for the transportation alternatives.
utilities = generate_utility_functions()

# Construct the log-likelihood of the multinomial logit model.
log_likelihood = loglogit(utilities, None, Choice)

# Create the Biogeme object used for estimation.
biogeme = BIOGEME(
    database,
    log_likelihood,
)
biogeme.model_name = 'plot_h01_mode_logit'

# Estimate the model, or reload the saved results if they are already available.
yaml_file_name = f'saved_results/{biogeme.model_name}.yaml'
results = biogeme.estimate_or_load(yaml_file_name=yaml_file_name)

# Display a compact summary and the estimated parameters.
print(results.short_summary())
print(get_pandas_estimated_parameters(estimation_results=results))

general_statistics = get_latex_general_statistics(estimation_results=results)
print(general_statistics)

estimated_parameters = get_latex_estimated_parameters(estimation_results=results)
print(estimated_parameters[''])
