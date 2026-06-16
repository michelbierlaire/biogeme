"""
Gaussian MIMIC model: maximum likelihood estimation
===================================================

This example estimates a Gaussian MIMIC model with one latent variable and no
choice component. The observed Likert indicators are treated as continuous
responses and modeled with Gaussian measurement equations.

The example introduces the latent-variable-only setting used later in the
hybrid-choice tutorial. The latent-variable structure itself is imported from a
separate semantic specification file. This script adds the Gaussian measurement
configuration, the normalization constraints required for identification, and
the maximum-likelihood estimation setup.

The script performs the following steps:

- load the latent-variable and indicator specifications,
- define a Gaussian measurement configuration for all indicators,
- define the normalization constraints used for identification,
- resolve the semantic specification into an estimable model,
- build the Biogeme expressions from the resolved model,
- integrate over the latent variable using Monte Carlo draws,
- construct the corresponding log-likelihood,
- estimate the model, or reload previously saved estimation results,
- display the estimated parameters as pandas and LaTeX tables.

Michel Bierlaire
Mon Jun 15 2026, 09:54:37
"""

from __future__ import annotations

from likert_spec import likert_indicators, likert_types
from number_of_draws import NUMBER_OF_DRAWS
from one_latent_variable_spec import DEFAULT_SIGMA_START, latent_variables
from optima import read_data

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import log
from biogeme.latent_variables import (
    BuildContext,
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
from biogeme.results_processing import (
    get_latex_estimated_parameters,
    get_latex_general_statistics,
    get_pandas_estimated_parameters,
)

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Gaussian measurement configuration for all Likert indicators.
measurement_configuration = MeasurementConfiguration(
    specifications=[
        IndicatorMeasurementSpec(
            indicator_name=indicator.name,
            measurement_model=MeasurementModel.GAUSSIAN,
            measurement_sigma=PositiveParameterSpec(start=DEFAULT_SIGMA_START),
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
# --------------------------------------------------
# A latent variable is not directly observed, so its location and scale are
# not identified unless we impose normalization constraints.
#
# We use a reference-indicator strategy based on indicator ``Envir01``:
#
# 1. The intercept of the reference indicator is fixed to 0.
#    This anchors the location of the latent-variable scale.
#
# 2. The loading of the reference indicator on the latent variable is fixed
#    to -1.
#    This anchors the scale of the latent variable and fixes its orientation.
#    The negative sign means that higher values of the latent variable imply
#    lower expected values for ``Envir01``.
#
# With these two constraints in place, the remaining intercepts and loadings
# are interpreted relative to the reference indicator.
normalization_plan = NormalizationPlan()

normalization_plan.add(
    Fixing(MeasurementIntercept('Envir01'), 0.0, note='reference indicator: location')
)
normalization_plan.add(
    Fixing(
        MeasurementLoading('car_centric_attitude', 'Envir01'),
        -1.0,
        note='reference indicator: scale and orientation',
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
# Biogeme for maximum-likelihood estimation.
built_model = build_biogeme_model(resolved_model)

# %%
# Log-likelihood
# --------------
# The builder provides the Monte-Carlo integrated measurement likelihood.
# Taking its logarithm yields the log-likelihood used for maximum-likelihood
# estimation.
integrated_likelihood = built_model.integrated_likelihood
log_likelihood = log(integrated_likelihood)

# %%
# Estimate the model with Biogeme.
# --------------------------------
# Existing results are reloaded from the YAML file when available.
biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=NUMBER_OF_DRAWS,
    calculating_second_derivatives='never',
    numerically_safe=False,
    max_iterations=5_000,
    group_of_parameters=built_model.parameter_groups,
)
biogeme.model_name = 'plot_h02_lv_mimic_gauss'

yaml_file_name = f'saved_results/{biogeme.model_name}.yaml'
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
