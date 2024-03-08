"""

biogeme.biogeme
===============

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Thu Nov 16 18:36:35 2023
"""

import biogeme.version as ver
import biogeme.biogeme as bio
import biogeme.database as db
import pandas as pd
from biogeme.expressions import Beta, Variable, exp
import biogeme.biogeme_logging as blog
from biogeme.function_output import BiogemeFunctionOutput

# %%
# Version of Biogeme.
print(ver.get_text())


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
my_data = db.Database('test', df)

# %%
# Data
my_data.data


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
my_biogeme = bio.BIOGEME(my_data, dict_of_expressions)
my_biogeme.modelName = 'simple_example'
print(my_biogeme)

# %%
# The data is stored in the Biogeme object.
my_biogeme.database.data

# %%
# Log likelihood with the initial values of the parameters.
my_biogeme.calculate_init_likelihood()

# %%
# Calculate the log-likelihood with a different value of the
# parameters. We retieve the current value and add 1 to each of them.
x = my_biogeme.id_manager.free_betas_values
xplus = [v + 1 for v in x]
print(xplus)

# %%
my_biogeme.calculate_likelihood(xplus, scaled=True)

# %%
# Calculate the log-likelihood function and its derivatives.
the_function_output: BiogemeFunctionOutput = (
    my_biogeme.calculate_likelihood_and_derivatives(
        xplus, scaled=True, hessian=True, bhhh=True
    )
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
# Now the unscaled version.
f, g, h, bhhh = my_biogeme.calculateLikelihoodAndDerivatives(
    xplus, scaled=False, hessian=True, bhhh=True
)
# %%
print(f'f = {f}')

# %%
print(f'g = {g}')

# %%
pd.DataFrame(h)

# %%
pd.DataFrame(bhhh)


# %%
# Calculate the hessian of the log likelihood function using finite difference.
fin_diff_hessian = my_biogeme.likelihood_finite_difference_hessian(xplus)
pd.DataFrame(fin_diff_hessian)

# %%
# Check numerically the derivatives' implementation. The analytical
# derivatives are compared to the numerical derivatives obtains by
# finite differences.
f, g, h, gdiff, hdiff = my_biogeme.check_derivatives(xplus, verbose=True)

# %%
print(f'f = {f}')
# %%
print(f'g = {g}')
# %%
pd.DataFrame(h)
# %%
pd.DataFrame(gdiff)
# print(f'gdiff = {gdiff}')
# %%
pd.DataFrame(hdiff)
# print(f'hdiff = {hdiff}')

# %%
# Estimation
# ----------

# %%
# Estimation of the parameters, with bootstrapping
my_biogeme.bootstrap_samples = 10
results = my_biogeme.estimate(run_bootstrap=True)

# %%
results.get_estimated_parameters()

# %%
# If the model has already been estimated, it is possible to recycle
# the estimation results. In that case, the other arguments are
# ignored, and the results are whatever is in the file.

# %%
recycled_results = my_biogeme.estimate(recycle=True, run_bootstrap=True)

# %%
print(recycled_results.short_summary())

# %%
recycled_results.get_estimated_parameters()

# %%
# Simulation
# ----------

# %%
# Simulate with the initial values for the parameters.
simulation_with_default_betas = my_biogeme.simulate(
    my_biogeme.log_like.get_beta_values()
)
simulation_with_default_betas

# %%
# Simulate with the estimated values for the parameters.

# %%
print(results.get_beta_values())

# %%
simulation_with_estimated_betas = my_biogeme.simulate(results.get_beta_values())
simulation_with_estimated_betas

# %%
# Confidence intervals.
# First, we extract the values of betas from the bootstrapping draws.
draws_from_betas = results.get_betas_for_sensitivity_analysis(
    my_biogeme.id_manager.free_betas.names
)
for draw in draws_from_betas:
    print(draw)

# %%
# Then, we calculate the confidence intervals. The default interval
# size is 0.9. Here, we use a different one.
left, right = my_biogeme.confidence_intervals(draws_from_betas, interval_size=0.95)
left

# %%
right

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
validationData = my_data.split(slices=5)
validation_results = my_biogeme.validate(results, validationData)

# %%
for slide in validation_results:
    print(
        f'Log likelihood for {slide.shape[0]} '
        f'validation data: {slide["Loglikelihood"].sum()}'
    )


# %%
# The following tools is used to find .py with the model name and a
# specific extension.
my_biogeme.files_of_type('pickle')
