"""
Gaussian hybrid mode choice model: simultaneous maximum likelihood estimation
============================================================================

This example estimates a hybrid mode choice model with one latent variable and
Gaussian measurement equations. The observed Likert indicators are treated as
continuous responses and modeled with Gaussian measurement equations.

In contrast with the sequential specification introduced previously, the
measurement component and the mode-choice component are estimated jointly. The
conditional likelihood of the indicators and the conditional likelihood of the
observed choice are multiplied conditionally on the latent variable, and the
resulting expression is integrated over the latent-variable distribution.

The latent-variable structure is imported from a separate semantic specification
file. This script adds the Gaussian measurement configuration, the normalization
constraints required for identification, the mode-choice utilities, and the
simultaneous maximum-likelihood estimation setup.

The script performs the following steps:

- load the latent-variable and indicator specifications,
- define a Gaussian measurement configuration for all indicators,
- define the normalization constraints used for identification,
- resolve the semantic specification into an estimable model,
- build the Biogeme expressions from the resolved model,
- build the choice utilities, including the latent-variable term,
- combine the measurement and choice conditional likelihoods,
- integrate the combined conditional likelihood over the latent variable,
- estimate the hybrid model, or reload previously saved estimation results,
- display the estimated parameters as grouped pandas and LaTeX tables.

Michel Bierlaire
Mon Jun 15 2026, 09:54:37
"""

from __future__ import annotations

import os
from pathlib import Path

from choice_latent_variables import generate_utility_functions
from likert_spec import likert_indicators, likert_types
from number_of_draws import NUMBER_OF_DRAWS
from one_latent_variable_spec import latent_variables
from optima import Choice, read_data

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import MonteCarlo, log
from biogeme.latent_variables import (
    BuildContext,
    BuiltBiogemeModel,
    EstimationMode,
    Fixing,
    IndicatorMeasurementSpec,
    MeasurementConfiguration,
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementModel,
    NormalizationPlan,
    PositiveParameterSpec,
    PositivityMode,
    build_biogeme_model,
    resolve_model,
)
from biogeme.models import logit
from biogeme.results_processing import (
    get_latex_estimated_parameters,
    get_latex_general_statistics,
    get_pandas_estimated_parameters,
)

logger = blog.get_screen_logger(level=blog.INFO)

DEFAULT_MEASUREMENT_SIGMA_START = 10.0

# %%
# Gaussian measurement configuration for all Likert indicators.
# -------------------------------------------------------------
measurement_configuration = MeasurementConfiguration(
    specifications=[
        IndicatorMeasurementSpec(
            indicator_name=indicator.name,
            measurement_model=MeasurementModel.GAUSSIAN,
            measurement_sigma=PositiveParameterSpec(
                start=DEFAULT_MEASUREMENT_SIGMA_START
            ),
        )
        for indicator in likert_indicators
    ]
)

# %%
# Load the Optima data.
# ---------------------
database = read_data()

# %%
# Define the build context for maximum likelihood estimation.
# -----------------------------------------------------------
# The build context specifies how the semantic latent-variable specification is
# translated into estimable Biogeme expressions. We start from the default
# maximum-likelihood context and explicitly select the log-exp parameterization
# for positive parameters.
default_context = BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD)
context = BuildContext(
    estimation_mode=default_context.estimation_mode,
    draw_type=default_context.draw_type,
    positivity_mode=PositivityMode.LOG_EXP,
    naming=default_context.naming,
    ordinal_eps=default_context.ordinal_eps,
    ordinal_enforce_order=default_context.ordinal_enforce_order,
)

# %%
# Identification constraints for the latent variable
# ---------------------------------------------------
# A latent variable is not directly observed, so its location and scale are
# not identified unless normalization constraints are imposed.
#
# We use a reference-indicator strategy based on ``Envir01``:
# - the intercept of ``Envir01`` is fixed to 0.0, which anchors the location
#   of the latent-variable scale;
# - the loading of ``Envir01`` on ``car_centric_attitude`` is fixed to -1.0,
#   which anchors the scale and fixes the orientation of the latent variable.
#
# The negative sign means that higher values of ``car_centric_attitude`` imply
# lower expected values for ``Envir01``. With these constraints in place, the
# remaining intercepts and loadings are interpreted relative to the reference
# indicator.
normalization_plan = NormalizationPlan()

normalization_plan.add(
    Fixing(
        MeasurementIntercept('Envir01'),
        0.0,
        note='car_centric_attitude reference indicator: location',
    )
)
normalization_plan.add(
    Fixing(
        MeasurementLoading('car_centric_attitude', 'Envir01'),
        -1.0,
        note='car_centric_attitude reference indicator: scale and orientation',
    )
)

# %%
# Resolve the semantic specification.
# -----------------------------------
# The resolver combines the latent-variable specification, the indicator
# definitions, the Gaussian measurement configuration, and the normalization plan
# into an internal resolved model.
resolved_model = resolve_model(
    latent_variables=latent_variables,
    likert_indicators=likert_indicators,
    likert_types=likert_types,
    measurement_configuration=measurement_configuration,
    context=context,
    normalization_plan=normalization_plan,
)

# %%
# Build the Biogeme expressions.
# ------------------------------
# The builder translates the resolved model into expressions that can be used by
# Biogeme for simultaneous maximum-likelihood estimation. It also provides
# report-ready parameter groups based on the parameters that are actually
# estimated.
built_model: BuiltBiogemeModel = build_biogeme_model(resolved_model)

# %%
# Choice utilities including the latent-variable term.
# ----------------------------------------------------
utilities = generate_utility_functions(built_model.latent_expressions)

# %%
# Conditional likelihood of the mode choice model
# -----------------------------------------------
conditional_choice_likelihood = logit(utilities, None, Choice)

# %%
# Combined conditional likelihood
# -------------------------------
# The measurement component and the mode-choice component are combined
# conditionally on the latent variable. This is the key difference with the
# sequential approach: both parts are estimated jointly in a single likelihood.
combined_conditional_likelihood = (
    built_model.conditional_likelihood * conditional_choice_likelihood
)

# %%
# Log-likelihood
# --------------
# The combined conditional likelihood is integrated over the latent variable
# by Monte Carlo. Taking the logarithm of the resulting integrated likelihood
# yields the log-likelihood used for simultaneous maximum-likelihood
# estimation.
integrated_likelihood = MonteCarlo(combined_conditional_likelihood)
log_likelihood = log(integrated_likelihood)

# %%
# Estimate the model with Biogeme.
# --------------------------------
# Existing results are reloaded from the YAML file when available.
number_of_draws = int(os.environ.get('BIOGEME_H04_NUMBER_OF_DRAWS', NUMBER_OF_DRAWS))
max_iterations = int(os.environ.get('BIOGEME_H04_MAX_ITERATIONS', '5000'))

biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=number_of_draws,
    calculating_second_derivatives='analytical',
    analytical_hessian_mode='chunked',
    hessian_parameter_block_size=4,
    hessian_observation_batch_size=100,
    max_iterations=max_iterations,
    group_of_parameters=built_model.parameter_groups,
)
biogeme.model_name = 'plot_h04_mode_lv_gauss_simult'

results_directory = Path(
    os.environ.get('BIOGEME_H04_RESULTS_DIRECTORY', 'saved_results')
)
results_directory.mkdir(parents=True, exist_ok=True)
yaml_file_name = results_directory / f'{biogeme.model_name}.yaml'
results = biogeme.estimate_or_load(yaml_file_name=yaml_file_name)

# %%
# Display a compact summary and the estimated parameters.
print(results.short_summary())
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
    group_of_parameters=built_model.parameter_groups,
)
for group_name, pandas_table in pandas_results.items():
    print(group_name if group_name else 'Estimated parameters')
    print(pandas_table)

general_statistics = get_latex_general_statistics(estimation_results=results)
print(general_statistics)

estimated_parameters = get_latex_estimated_parameters(
    estimation_results=results,
    group_of_parameters=built_model.parameter_groups,
)
for group_name, latex_table in estimated_parameters.items():
    print(group_name if group_name else 'Estimated parameters')
    print(latex_table)
