"""
Ordered-logit hybrid mode choice model: simultaneous maximum likelihood estimation
==================================================================================

This example estimates a hybrid mode choice model with one latent variable and
ordered-logit measurement equations. The observed Likert indicators are modeled
as ordinal responses, and the associated threshold parameters are estimated
jointly with the latent-variable and choice-model parameters.

Compared with the previous Gaussian specification, only the measurement model
changes: the hybrid structure, the latent variable, and the simultaneous
maximum-likelihood estimation strategy remain the same.

The latent-variable structure is imported from a separate semantic specification
file. This script adds the ordered-logit measurement configuration, the
normalization constraints required for identification, the mode-choice utilities,
and the simultaneous maximum-likelihood estimation setup.

The script performs the following steps:

- load the latent-variable and indicator specifications,
- define an ordered-logit measurement configuration for all indicators,
- define the normalization constraints used for identification,
- resolve the semantic specification into an estimable model,
- build the Biogeme expressions from the resolved model,
- build the choice utilities, including the latent-variable term,
- combine the ordered-logit measurement and choice conditional likelihoods,
- integrate the combined conditional likelihood over the latent variable,
- estimate the hybrid model, or reload previously saved estimation results,
- display the estimated parameters as grouped pandas and LaTeX tables.

Michel Bierlaire
Mon Jun 15 2026, 09:54:37
"""

from __future__ import annotations

from choice_latent_variables import generate_availability, generate_utility_functions
from likert_spec import likert_indicators, likert_types
from number_of_draws import NUMBER_OF_DRAWS
from one_latent_variable_spec import latent_variables
from optima import Choice, read_data

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import MonteCarlo, log
from biogeme.latent_variables import (
    BuildContext,
    EstimationMode,
    Fixing,
    IndicatorMeasurementSpec,
    MeasurementConfiguration,
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementModel,
    MeasurementSigma,
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
# Ordered-logit measurement configuration for all Likert indicators.
# ------------------------------------------------------------------
measurement_configuration = MeasurementConfiguration(
    specifications=[
        IndicatorMeasurementSpec(
            indicator_name=indicator.name,
            measurement_model=MeasurementModel.ORDERED_LOGIT,
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
# for positive parameters. The ordinal options are relevant here because the
# measurement equations use ordered-logit probabilities.
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
# Two layers of normalization are needed in this ordered-logit specification.
#
# 1. Ordered-logit measurement-scale normalization
#    The latent response underlying the ordinal indicators has an arbitrary
#    scale. We therefore fix one measurement sigma to 1.0 in order to anchor
#    the overall scale of the ordered-logit measurement model.
#
# 2. Latent-variable normalization by reference indicator
#    The latent variable itself is not directly observed, so its location and
#    orientation must also be fixed. We use indicator ``Envir01`` as the
#    reference indicator:
#
#    - its intercept is fixed to 0.0, which anchors the location of the latent
#      variable;
#    - its loading on ``car_centric_attitude`` is fixed to -1.0, which anchors
#      the scale and fixes the orientation of the latent variable.
#
# The negative sign means that higher values of ``car_centric_attitude`` imply
# lower expected values for ``Envir01``. With these constraints in place, the
# remaining intercepts, loadings, and thresholds are interpreted relative to
# the chosen reference and measurement scale.
normalization_plan = NormalizationPlan()

normalization_plan.add(
    Fixing(
        MeasurementSigma('Envir01'),
        1.0,
        note='ordered-logit measurement model: scale normalization',
    )
)

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
# definitions, the ordered-logit measurement configuration, and the normalization
# plan into an internal resolved model.
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
built_model = build_biogeme_model(resolved_model)

# %%
# Choice utilities including the latent-variable term.
# ----------------------------------------------------
utilities = generate_utility_functions(built_model.latent_expressions)

# %%
# Conditional likelihood of the mode choice model
# -----------------------------------------------

availability = generate_availability()
conditional_choice_likelihood = logit(utilities, availability, Choice)

# %%
# Combined conditional likelihood
# -------------------------------
# The ordered-logit measurement component and the mode-choice component are
# combined conditionally on the latent variable. As in the previous step,
# both parts are estimated jointly in a single likelihood.
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
biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=NUMBER_OF_DRAWS,
    calculating_second_derivatives='never',
    max_iterations=5_000,
    group_of_parameters=built_model.parameter_groups,
)
biogeme.model_name = 'plot_h05_mode_lv_ordlogit_simult'

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
