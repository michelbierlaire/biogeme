"""
Sequential estimation of a choice model with a latent variable
==============================================================

This example illustrates the second step of a sequential estimation
procedure.

The latent variable and its structural equation are first estimated using the
Gaussian MIMIC model of ``plot_h02_lv_mimic_gauss``. The estimated structural
parameters are then loaded from the saved results file and used to construct
the latent-variable expression appearing in the choice model.

Only the choice-model parameters are estimated in this script. The latent
variable is not re-estimated jointly with the measurement equations.

Michel Bierlaire
Mon Jun 15 2026, 09:54:37
"""

import sys

from choice_latent_variables import generate_utility_functions
from number_of_draws import NUMBER_OF_DRAWS
from one_latent_variable_spec import car_centric_attitude
from optima import (
    Choice,
    read_data,
)

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Draws,
    MonteCarlo,
    MultipleSum,
    Variable,
    exp,
    log,
)
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_latex_estimated_parameters,
    get_latex_general_statistics,
    get_pandas_estimated_parameters,
)

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Load data
database = read_data()

# %%
# Read the results of the previously estimated MIMIC model.
FILE_WITH_ESTIMATION_RESULTS = 'saved_results/plot_h02_lv_mimic_gauss.yaml'
try:
    estimation_results = EstimationResults.from_yaml_file(
        filename=FILE_WITH_ESTIMATION_RESULTS
    )
except FileNotFoundError:
    print(
        f'File {FILE_WITH_ESTIMATION_RESULTS} does not exist. Sequential estimation cannot be performed.'
    )
    sys.exit()
estimated_parameters: dict[str, float] = estimation_results.get_beta_values()

# %%
# Reconstruct the latent-variable expression using the parameter estimates
# obtained from the MIMIC model. The sequential specification keeps both the
# deterministic component and the random structural disturbance.
list_of_explanatory_variables = (
    car_centric_attitude.structural_equation.explanatory_variables
)

# The sigma parameter was estimated on the log scale in the MIMIC model.
# We therefore exponentiate it before scaling the structural disturbance.
terms = [
    estimated_parameters[f'struct_car_centric_attitude_{var}'] * Variable(var)
    for var in list_of_explanatory_variables
]

structural_sigma = exp(estimated_parameters['struct_car_centric_attitude_sigma_log'])
structural_error = structural_sigma * Draws(
    'car_centric_attitude_draw', 'NORMAL_MLHS_ANTI'
)

car_centric_attitude = MultipleSum(terms) + structural_error
latent_expressions = {'car_centric_attitude': car_centric_attitude}

# %%
# Utility functions including the latent-variable effect.
v = generate_utility_functions(latent_expressions=latent_expressions)

# %%
# Choice model estimated conditionally on the latent-variable specification.
conditional_likelihood = logit(v, None, Choice)

loglikelihood = log(MonteCarlo(conditional_likelihood))

biogeme = BIOGEME(
    database,
    loglikelihood,
    number_of_draws=NUMBER_OF_DRAWS,
    numerically_safe=True,
    analytical_hessian_mode='chunked',
    hessian_parameter_block_size=4,
    hessian_observation_batch_size=100,
    max_iterations=5_000,
)
biogeme.model_name = 'plot_h03_model_lv_gauss_seq'

yaml_file_name = f'saved_results/{biogeme.model_name}.yaml'
results = biogeme.estimate_or_load(yaml_file_name=yaml_file_name)

print(results.short_summary())
print(get_pandas_estimated_parameters(estimation_results=results))

general_statistics = get_latex_general_statistics(estimation_results=results)
print(general_statistics)

estimated_parameters = get_latex_estimated_parameters(estimation_results=results)
for group_name, latex_table in estimated_parameters.items():
    print(group_name if group_name else 'Estimated parameters')
    print(latex_table)
