"""

biogeme.biogeme
===============

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

Michel Bierlaire
Sun Jun 29 2025, 01:14:48
"""

import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.calculator import evaluate_formula
from biogeme.database import Database
from biogeme.expressions import Beta, Variable, exp
from biogeme.function_output import FunctionOutput
from biogeme.results_processing import get_pandas_estimated_parameters
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.tools import CheckDerivativesResults
from biogeme.tools.files import files_of_type
from biogeme.validation import ValidationResult
from biogeme.version import get_text

# %%
# Version of Biogeme.
print(get_text())


# %%
# Logger.
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Logger initialized')


# %%
# Definition of a database
df = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [1, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)
my_data = Database('test', df)

# %%
# Data
display(my_data.dataframe)


# %%
# Definition of various expressions.
Variable1 = Variable('Variable1')
Variable2 = Variable('Variable2')
beta1 = Beta('beta1', -1.0, -3, 3, 0)
beta2 = Beta('beta2', 2.0, -3, 10, 0)
likelihood = -(beta1**2) * Variable1 - exp(beta2 * beta1) * Variable2 - beta2**4
simul = beta1 / Variable1 + beta2 / Variable2
dict_of_expressions = {'log_like': likelihood, 'beta1': beta1, 'simul': simul}

# %%
# Creation of the BIOGEME object.
my_biogeme = BIOGEME(my_data, dict_of_expressions)
my_biogeme.model_name = 'simple_example'
print(my_biogeme)

# %%
# The data is stored in the Biogeme object.
display(my_biogeme.database.dataframe)

# %%
# Log likelihood with the initial values of the parameters.
my_biogeme.calculate_init_likelihood()

# %%
# Calculate the log-likelihood with a different value of the
# parameters. We retrieve the current value and add 1 to each of them.
x = my_biogeme.expressions_registry.free_betas_init_values
x_plus = {key: value + 1.0 for key, value in x.items()}
print(x_plus)

# %%
log_likelihood_x_plus = evaluate_formula(
    model_elements=my_biogeme.model_elements,
    the_betas=x_plus,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    numerically_safe=False,
)
print(log_likelihood_x_plus)

# %%
# Calculate the log-likelihood function and its derivatives.
the_function_output: FunctionOutput = my_biogeme.function_evaluator.evaluate(
    the_betas=x_plus,
    gradient=True,
    hessian=True,
    bhhh=True,
)

# %%
print(f'f = {the_function_output.function}')

# %%
print(f'g = {the_function_output.gradient}')

# %%
pd.DataFrame(the_function_output.hessian)

# %%
pd.DataFrame(the_function_output.bhhh)


# %%
# Check numerically the derivatives' implementation. The analytical
# derivatives are compared to the numerical derivatives obtains by
# finite differences.
check_results: CheckDerivativesResults = my_biogeme.check_derivatives(verbose=True)

# %%
print(f'f = {check_results.function}')
# %%
print(f'g = {check_results.analytical_gradient}')
# %%
display(pd.DataFrame(check_results.analytical_hessian))
# %%
display(pd.DataFrame(check_results.finite_differences_gradient))
# %%
display(pd.DataFrame(check_results.finite_differences_hessian))

# %%
# Estimation
# ----------

# %%
# Estimation of the parameters, with bootstrapping
results = my_biogeme.estimate(run_bootstrap=True)

# %%
estimated_parameters = get_pandas_estimated_parameters(estimation_results=results)
display(estimated_parameters)

# %%
# If the model has already been estimated, it is possible to recycle
# the estimation results. In that case, the other arguments are
# ignored, and the results are whatever is in the file.

# %%
recycled_results = my_biogeme.estimate(recycle=True, run_bootstrap=True)

# %%
print(recycled_results.short_summary())

# %%
recycled_parameters = get_pandas_estimated_parameters(
    estimation_results=recycled_results
)
display(recycled_parameters)


# %%
# Simulation
# ----------

# %%
# Simulate with the estimated values for the parameters.

# %%
display(results.get_beta_values())

# %%
simulation_with_estimated_betas = my_biogeme.simulate(results.get_beta_values())
display(simulation_with_estimated_betas)

# %%
# Confidence intervals.
# First, we extract the values of betas from the bootstrapping draws.
draws_from_betas = results.get_betas_for_sensitivity_analysis()
for draw in draws_from_betas:
    print(draw)

# %%
# Then, we calculate the confidence intervals. The default interval
# size is 0.9. Here, we use a different one.
left, right = my_biogeme.confidence_intervals(draws_from_betas, interval_size=0.95)
display(left)

# %%
display(right)

# %%
# Validation
# ----------
# The validation consists in organizing the data into several slices
# of about the same size, randomly defined.  Each slide is considered
# as a validation dataset. The model is then re-estimated using all
# the data except the slice, and the estimated model is applied on the
# validation set (i.e. the slice). The value of the log likelihood for
# each observation in the validation set is reported in a
# dataframe. As this is done for each slice, the output is a list of
# dataframes, each corresponding to one of these exercises.

# %%
validation_results: list[ValidationResult] = my_biogeme.validate(results, slices=5)

# %%
for one_validation_result in validation_results:
    print(
        f'Log likelihood for {one_validation_result.validation_modeling_elements.sample_size} '
        f'validation data: {one_validation_result.simulated_values.iloc[0].sum()}'
    )


# %%
# The following tools is used to obtain the list of files with a given extension in the
# local directory.
display(files_of_type(extension='yaml', name=my_biogeme.model_name))
