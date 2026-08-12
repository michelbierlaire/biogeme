""".. _plot_h08_mode_lv_parameter_overrides:

8. Build-only hybrid-choice example with explicit parameter overrides
======================================================================

This example follows the simultaneous Gaussian hybrid mode-choice model in
``plot_h04_mode_lv_gauss_simult.py``.  It deliberately stops before creating a
``BIOGEME`` estimation object and therefore does not estimate anything.

The purpose is to show where overrides belong in a hybrid-choice workflow:
the semantic specification is resolved, the latent-variable builder creates
the generated Biogeme parameters, and only then is the final likelihood
rewritten.  Here, one generated measurement loading is fixed by replacing it
with a fixed ``Beta`` having the same name.

Michel Bierlaire, EPFL
"""

from __future__ import annotations

from choice_latent_variables import generate_availability, generate_utility_functions
from likert_spec import likert_indicators, likert_types
from one_latent_variable_spec import latent_variables
from optima import Choice

import biogeme.biogeme_logging as blog
from biogeme.expressions import (
    Beta,
    MonteCarlo,
    ParameterOverrides,
    apply_parameter_overrides,
    list_of_all_betas_in_expression,
    log,
)
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
from biogeme.models import logit

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example plot_h08_mode_lv_parameter_overrides.py')

# %%
# Define the Gaussian measurement configuration used by H04.
measurement_configuration = MeasurementConfiguration(
    specifications=[
        IndicatorMeasurementSpec(
            indicator_name=indicator.name,
            measurement_model=MeasurementModel.GAUSSIAN,
            measurement_sigma=PositiveParameterSpec(start=10.0),
        )
        for indicator in likert_indicators
    ]
)

# %%
# Resolve the semantic latent-variable specification.  The reference indicator
# fixes the location and scale of the latent variable.
default_context = BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD)
context = BuildContext(
    estimation_mode=default_context.estimation_mode,
    draw_type=default_context.draw_type,
    positivity_mode=PositivityMode.LOG_EXP,
    naming=default_context.naming,
    ordinal_eps=default_context.ordinal_eps,
    ordinal_enforce_order=default_context.ordinal_enforce_order,
)

normalization_plan = NormalizationPlan()
normalization_plan.add(
    Fixing(
        MeasurementIntercept('Envir01'),
        0.0,
        note='reference indicator: location',
    )
)
normalization_plan.add(
    Fixing(
        MeasurementLoading('car_centric_attitude', 'Envir01'),
        -1.0,
        note='reference indicator: scale and orientation',
    )
)

resolved_model = resolve_model(
    latent_variables=latent_variables,
    likert_indicators=likert_indicators,
    likert_types=likert_types,
    measurement_configuration=measurement_configuration,
    context=context,
    normalization_plan=normalization_plan,
)

# %%
# Build the expressions.  At this point the builder has created names such as
# ``measurement_coefficient_car_centric_attitude_Envir02`` automatically.
built_model = build_biogeme_model(resolved_model)
utilities = generate_utility_functions(built_model.latent_expressions)
availability = generate_availability()
conditional_choice_likelihood = logit(utilities, availability, Choice)
combined_conditional_likelihood = (
    built_model.conditional_likelihood * conditional_choice_likelihood
)

# %%
# Fix one automatically generated measurement loading.  The replacement uses
# the exact generated Beta name, so the parameter remains visible as a fixed
# coefficient while no estimation is performed in this example.
loading_name = 'measurement_coefficient_car_centric_attitude_Envir02'
overrides = ParameterOverrides()
overrides.set(loading_name, Beta(loading_name, 0.5, None, None, 1))

before = {
    beta.name
    for beta in list_of_all_betas_in_expression(combined_conditional_likelihood)
}
overridden_conditional_likelihood = apply_parameter_overrides(
    combined_conditional_likelihood, overrides
)
after = {
    beta.name
    for beta in list_of_all_betas_in_expression(overridden_conditional_likelihood)
}
overridden_loading = next(
    beta
    for beta in list_of_all_betas_in_expression(overridden_conditional_likelihood)
    if beta.name == loading_name
)

# Construct the final integrated likelihood only to demonstrate that the
# rewritten expression is ready for estimation.  No database, BIOGEME object,
# compilation, or estimation is performed.
log_likelihood = log(MonteCarlo(overridden_conditional_likelihood))

print(f'Overridden parameter: {loading_name}')
print(f'Parameter present before override: {loading_name in before}')
print(f'Parameter present after override: {loading_name in after}')
print(f'Overridden parameter status: {overridden_loading.status}')
print(f'Number of Betas before override: {len(before)}')
print(f'Number of Betas after override: {len(after)}')
print(f'Final expression type: {type(log_likelihood).__name__}')
