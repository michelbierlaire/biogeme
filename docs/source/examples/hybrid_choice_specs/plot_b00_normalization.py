"""
Resolve several latent-variable measurement specifications.

This script is an illustration of specification resolution only. It does not
estimate any model and does not use data. Instead, it builds a sequence of
latent-variable measurement specifications and sends each of them to
``resolve_model``. The resolver validates the specification, constructs the
internal ``ResolvedModel`` representation, determines the ordinal threshold
systems, and reports the normalization rules and warnings associated with the
specification.

For each resolved specification, the script prints:

- the number of latent variables,
- the number of indicators,
- the number of ordinal threshold systems,
- the measurement models used by the indicators,
- the threshold systems and the indicators attached to them,
- the normalization rules selected by the resolver,
- the warnings generated during resolution.

The cases include symmetric and non-symmetric ordinal threshold systems,
Gaussian measurement specifications, indicators shared by several latent
variables, unused threshold systems, latent variables without indicators, and
inconsistent specifications detected by the resolver.

Michel Bierlaire
Sat Jun 06 2026, 11:53:24
"""

from __future__ import annotations

from biogeme.latent_variables import (
    BuildContext,
    EstimationMode,
    IndicatorMeasurementSpec,
    LatentVariable,
    LikertIndicator,
    LikertType,
    MeasurementConfiguration,
    MeasurementModel,
    NormalizationPlan,
    PositiveParameterSpec,
    StructuralEquation,
    resolve_model,
)


# -----------------------------------------------------------------------------
# Helper functions used to build measurement configurations and display summaries
# -----------------------------------------------------------------------------
def ordinal_measurement_configuration(
    indicators: list[LikertIndicator],
    model: MeasurementModel = MeasurementModel.ORDERED_PROBIT,
    sigma_start: float = 1.0,
) -> MeasurementConfiguration:
    """Create an ordinal measurement configuration for the given indicators.

    All indicators are assigned the same ordinal measurement model and the same
    initial value for the positive scale parameter of the measurement equation.
    """
    return MeasurementConfiguration(
        specifications=[
            IndicatorMeasurementSpec(
                indicator_name=indicator.name,
                measurement_model=model,
                measurement_sigma=PositiveParameterSpec(start=sigma_start),
            )
            for indicator in indicators
        ]
    )


def gaussian_measurement_configuration(
    indicators: list[LikertIndicator],
    sigma_start: float = 10.0,
) -> MeasurementConfiguration:
    """Create a Gaussian measurement configuration for the given indicators.

    All indicators are assigned the Gaussian measurement model and the same initial
    value for the positive scale parameter of the measurement equation.
    """
    return MeasurementConfiguration(
        specifications=[
            IndicatorMeasurementSpec(
                indicator_name=indicator.name,
                measurement_model=MeasurementModel.GAUSSIAN,
                measurement_sigma=PositiveParameterSpec(start=sigma_start),
            )
            for indicator in indicators
        ]
    )


def format_resolved_summary(resolved) -> str:
    """Return a compact textual description of a resolved specification.

    The summary contains the number of latent variables and indicators, the
    measurement specifications, the ordinal threshold systems, the normalization
    rules, and the warnings reported by the resolver. No estimation results are
    involved.
    """
    measurement_models = ', '.join(
        model.value for model in resolved.metadata.measurement_models_present
    )
    lines: list[str] = [
        'Resolved model summary',
        f'- latent variables: {resolved.metadata.n_latent_variables}',
        f'- indicators: {resolved.metadata.n_indicators}',
        f'- ordinal threshold systems: {resolved.metadata.n_threshold_systems}',
        f'- measurement models present: {measurement_models}',
    ]

    if resolved.threshold_systems:
        lines.append('- threshold systems:')
        for type_name, system in resolved.threshold_systems.items():
            used_by = ', '.join(system.used_by_indicators)
            lines.append(
                f'  * {type_name}: {system.construction_kind.value}, used by [{used_by}]'
            )
    else:
        lines.append('- no ordinal threshold systems')

    lines.append('Normalization')
    if resolved.normalization.rules:
        for rule in resolved.normalization.rules:
            lines.append(f'- {rule.reason}: {rule.target_name} = {rule.value}')
    else:
        lines.append('- no explicit normalization plan provided')

    if resolved.normalization.warnings:
        lines.append('Warnings')
        for warning in resolved.normalization.warnings:
            lines.append(f'- {warning}')

    return '\n'.join(lines)


def print_case(
    title: str,
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
    measurement_configuration: MeasurementConfiguration,
) -> None:
    """Resolve one specification and print its summary without estimating it."""
    line = '=' * len(title)
    print('\n' + line)
    print(title)
    print(line)
    context = BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD)
    resolved = resolve_model(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        measurement_configuration=measurement_configuration,
        context=context,
        normalization_plan=NormalizationPlan(),
    )
    print(format_resolved_summary(resolved))


# =============================================================================
# Case 1: one latent variable with three ordinal indicators
#
# The three environmental-attitude indicators use the same symmetric Likert
# type. The resolved model contains one symmetric threshold system shared by
# Envir01, Envir02, and Envir03.
# =============================================================================

lv_1 = LatentVariable(
    name='environmental_attitude',
    structural_equation=StructuralEquation(
        name='environmental_attitude',
        explanatory_variables=['income', 'age', 'education'],
    ),
    indicators={'Envir01', 'Envir02', 'Envir03'},
)

indicators_1 = [
    LikertIndicator(
        name='Envir01',
        statement='I am concerned about climate change.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Envir02',
        statement='Public transport should be improved even if taxes increase.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Envir03',
        statement='Green policies are important for the future.',
        type_name='likert_sym',
    ),
]

types_1 = [
    LikertType(
        type_name='likert_sym',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[-1],
    )
]

print_case(
    'Case 1: one latent variable, 3 indicators, symmetric thresholds',
    latent_variables=[lv_1],
    likert_indicators=indicators_1,
    likert_types=types_1,
    measurement_configuration=ordinal_measurement_configuration(indicators_1),
)

# =============================================================================
# Case 2: one latent variable with three ordinal indicators
#
# The three comfort indicators use the same non-symmetric Likert type. The
# resolved model contains one non-symmetric threshold system shared by Comfort01,
# Comfort02, and Comfort03.
# =============================================================================

lv_2 = LatentVariable(
    name='comfort_attitude',
    structural_equation=StructuralEquation(
        name='comfort_attitude',
        explanatory_variables=['income', 'car_owner'],
    ),
    indicators={'Comfort01', 'Comfort02', 'Comfort03'},
)

indicators_2 = [
    LikertIndicator(
        name='Comfort01',
        statement='A comfortable seat is very important.',
        type_name='likert_asym',
    ),
    LikertIndicator(
        name='Comfort02',
        statement='I prefer comfort over speed.',
        type_name='likert_asym',
    ),
    LikertIndicator(
        name='Comfort03',
        statement='I avoid crowded vehicles when possible.',
        type_name='likert_asym',
    ),
]

types_2 = [
    LikertType(
        type_name='likert_asym',
        symmetric=False,
        categories=[0, 1, 2, 3],
        neutral_labels=[-1],
    )
]

print_case(
    'Case 2: one latent variable, 3 indicators, non-symmetric thresholds',
    latent_variables=[lv_2],
    likert_indicators=indicators_2,
    likert_types=types_2,
    measurement_configuration=ordinal_measurement_configuration(indicators_2),
)

# =============================================================================
# Case 3: two latent variables with one shared ordinal indicator
#
# Env01 and Env02 measure environmental_attitude. Car01 and Car02 measure
# car_centric_attitude. Common01 is attached to both latent variables. All five
# indicators use the same symmetric Likert type.
# =============================================================================

lv_3a = LatentVariable(
    name='environmental_attitude',
    structural_equation=StructuralEquation(
        name='environmental_attitude',
        explanatory_variables=['income', 'education'],
    ),
    indicators={'Env01', 'Env02', 'Common01'},
)

lv_3b = LatentVariable(
    name='car_centric_attitude',
    structural_equation=StructuralEquation(
        name='car_centric_attitude',
        explanatory_variables=['income', 'children'],
    ),
    indicators={'Car01', 'Car02', 'Common01'},
)

indicators_3 = [
    LikertIndicator(
        name='Env01',
        statement='I am worried about pollution.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Env02',
        statement='Cities should reduce car traffic.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Car01',
        statement='A car is essential for daily life.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Car02',
        statement='I enjoy driving.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Common01',
        statement='Transport choices reflect my lifestyle.',
        type_name='likert_sym',
    ),
]

types_3 = [
    LikertType(
        type_name='likert_sym',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[-1],
    )
]

print_case(
    'Case 3: two latent variables, one common indicator, symmetric thresholds',
    latent_variables=[lv_3a, lv_3b],
    likert_indicators=indicators_3,
    likert_types=types_3,
    measurement_configuration=ordinal_measurement_configuration(indicators_3),
)

# =============================================================================
# Case 4: one latent variable with two ordinal scales
#
# Serv01 and Serv02 use a symmetric Likert type. Freq01 and Freq02 use a
# non-symmetric count type. The resolved model therefore contains two threshold
# systems, one for each type.
# =============================================================================

lv_4 = LatentVariable(
    name='service_attitude',
    structural_equation=StructuralEquation(
        name='service_attitude',
        explanatory_variables=['age', 'income'],
    ),
    indicators={'Serv01', 'Serv02', 'Freq01', 'Freq02'},
)

indicators_4 = [
    LikertIndicator(
        name='Serv01',
        statement='Service quality matters a lot.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Serv02',
        statement='Friendly staff improves my experience.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Freq01',
        statement='How many times per week do you travel?',
        type_name='count_asym',
    ),
    LikertIndicator(
        name='Freq02',
        statement='How many times per month do you use public transport?',
        type_name='count_asym',
    ),
]

types_4 = [
    LikertType(
        type_name='likert_sym',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[-1],
    ),
    LikertType(
        type_name='count_asym',
        symmetric=False,
        categories=[0, 1, 2, 3, 4],
        neutral_labels=[-1],
    ),
]

print_case(
    'Case 4: one latent variable, mixed symmetric and non-symmetric threshold systems',
    latent_variables=[lv_4],
    likert_indicators=indicators_4,
    likert_types=types_4,
    measurement_configuration=ordinal_measurement_configuration(indicators_4),
)

# =============================================================================
# Case 5: one latent variable with three Gaussian indicators
#
# Sat01, Sat02, and Sat03 are configured with Gaussian measurement models. The
# resolved model contains no ordinal threshold system for this case.
# =============================================================================

lv_5 = LatentVariable(
    name='satisfaction_attitude',
    structural_equation=StructuralEquation(
        name='satisfaction_attitude',
        explanatory_variables=['income', 'age', 'car_owner'],
    ),
    indicators={'Sat01', 'Sat02', 'Sat03'},
)

indicators_5 = [
    LikertIndicator(
        name='Sat01',
        statement='Overall, I am satisfied with the quality of the service.',
        type_name='likert_continuous',
    ),
    LikertIndicator(
        name='Sat02',
        statement='The service is reliable enough for my daily needs.',
        type_name='likert_continuous',
    ),
    LikertIndicator(
        name='Sat03',
        statement='I would recommend this service to friends or colleagues.',
        type_name='likert_continuous',
    ),
]

types_5 = [
    LikertType(
        type_name='likert_continuous',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[-1],
    )
]

print_case(
    'Case 5: one latent variable, 3 Gaussian indicators',
    latent_variables=[lv_5],
    likert_indicators=indicators_5,
    likert_types=types_5,
    measurement_configuration=gaussian_measurement_configuration(indicators_5),
)

# =============================================================================
# Case 6a: latent variable with no indicator
#
# The latent variable unobserved_attitude has no associated indicator. The
# resolver produces a warning and the summary displays it.
# =============================================================================

lv_6a = LatentVariable(
    name='unobserved_attitude',
    structural_equation=StructuralEquation(
        name='unobserved_attitude',
        explanatory_variables=['income'],
    ),
    indicators=set(),
)

indicators_6a: list[LikertIndicator] = []
types_6a: list[LikertType] = []

print_case(
    'Case 6a: warning - latent variable with no indicator',
    latent_variables=[lv_6a],
    likert_indicators=indicators_6a,
    likert_types=types_6a,
    measurement_configuration=ordinal_measurement_configuration(indicators_6a),
)

# =============================================================================
# Case 6b: declared threshold system used by no indicator
#
# The specification declares two Likert types. Only used_type is referenced by an
# indicator. unused_type is declared but unused, which is reported by the
# resolver.
# =============================================================================

lv_6b = LatentVariable(
    name='simple_attitude',
    structural_equation=StructuralEquation(
        name='simple_attitude',
        explanatory_variables=['income'],
    ),
    indicators={'Only01'},
)

indicators_6b = [
    LikertIndicator(
        name='Only01',
        statement='I like this service.',
        type_name='used_type',
    ),
]

types_6b = [
    LikertType(
        type_name='used_type',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[-1],
    ),
    LikertType(
        type_name='unused_type',
        symmetric=False,
        categories=[0, 1, 2],
        neutral_labels=[-1],
    ),
]

print_case(
    'Case 6b: warning - threshold system used by no indicator',
    latent_variables=[lv_6b],
    likert_indicators=indicators_6b,
    likert_types=types_6b,
    measurement_configuration=ordinal_measurement_configuration(indicators_6b),
)

# =============================================================================
# Case 6c: threshold system used by one indicator
#
# Rare01 is the only indicator that refers to rare_type. The resolver keeps the
# threshold system and reports that it is used by a single indicator.
# =============================================================================

lv_6c = LatentVariable(
    name='rare_attitude',
    structural_equation=StructuralEquation(
        name='rare_attitude',
        explanatory_variables=['education'],
    ),
    indicators={'Rare01'},
)

indicators_6c = [
    LikertIndicator(
        name='Rare01',
        statement='This is a rare indicator.',
        type_name='rare_type',
    ),
]

types_6c = [
    LikertType(
        type_name='rare_type',
        symmetric=False,
        categories=[0, 1, 2, 3],
        neutral_labels=[-1],
    ),
]

print_case(
    'Case 6c: warning - threshold system with only one indicator',
    latent_variables=[lv_6c],
    likert_indicators=indicators_6c,
    likert_types=types_6c,
    measurement_configuration=ordinal_measurement_configuration(indicators_6c),
)

# =============================================================================
# Case 6d: two latent variables with the same indicators
#
# attitude_a and attitude_b are both associated with Shared01 and Shared02. The
# resolver reports that neither latent variable has an indicator specific to it.
# =============================================================================

lv_6d_a = LatentVariable(
    name='attitude_a',
    structural_equation=StructuralEquation(
        name='attitude_a',
        explanatory_variables=['income'],
    ),
    indicators={'Shared01', 'Shared02'},
)

lv_6d_b = LatentVariable(
    name='attitude_b',
    structural_equation=StructuralEquation(
        name='attitude_b',
        explanatory_variables=['age'],
    ),
    indicators={'Shared01', 'Shared02'},
)

indicators_6d = [
    LikertIndicator(
        name='Shared01',
        statement='Shared indicator 1.',
        type_name='likert_sym',
    ),
    LikertIndicator(
        name='Shared02',
        statement='Shared indicator 2.',
        type_name='likert_sym',
    ),
]

types_6d = [
    LikertType(
        type_name='likert_sym',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[-1],
    ),
]

print_case(
    'Case 6d: warning - no unique indicator for either latent variable',
    latent_variables=[lv_6d_a, lv_6d_b],
    likert_indicators=indicators_6d,
    likert_types=types_6d,
    measurement_configuration=ordinal_measurement_configuration(indicators_6d),
)

# =============================================================================
# Case 6e: indicator referenced by a latent variable but not defined
#
# broken_attitude refers to Missing01 and Defined01, but only Defined01 is
# declared in the list of indicators. The resolver call is wrapped in a
# try-except block so that the detected problem is printed.
# =============================================================================

print('\n' + '=' * 63)
print('Case 6e: problem - indicator referenced by a latent variable but not defined')
print('=' * 63)

lv_6e = LatentVariable(
    name='broken_attitude',
    structural_equation=StructuralEquation(
        name='broken_attitude',
        explanatory_variables=['income'],
    ),
    indicators={'Missing01', 'Defined01'},
)

indicators_6e = [
    LikertIndicator(
        name='Defined01',
        statement='Defined indicator.',
        type_name='likert_sym',
    ),
]

types_6e = [
    LikertType(
        type_name='likert_sym',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[-1],
    ),
]

try:
    resolved_6e = resolve_model(
        latent_variables=[lv_6e],
        likert_indicators=indicators_6e,
        likert_types=types_6e,
        measurement_configuration=ordinal_measurement_configuration(indicators_6e),
        context=BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD),
        normalization_plan=NormalizationPlan(),
    )
    print(format_resolved_summary(resolved_6e))
except Exception as e:
    print(f'Expected problem detected: {e}')

# =============================================================================
# Case 6f: indicator referring to an unknown Likert type
#
# Broken01 refers to unknown_type, but the specification does not declare any
# LikertType with that name. The resolver call is wrapped in a try-except block
# so that the detected problem is printed.
# =============================================================================

print('\n' + '=' * 59)
print('Case 6f: problem - indicator refers to an unknown threshold system')
print('=' * 59)

lv_6f = LatentVariable(
    name='broken_type_attitude',
    structural_equation=StructuralEquation(
        name='broken_type_attitude',
        explanatory_variables=['income'],
    ),
    indicators={'Broken01'},
)

indicators_6f = [
    LikertIndicator(
        name='Broken01',
        statement='Indicator with missing type.',
        type_name='unknown_type',
    ),
]

types_6f: list[LikertType] = []

try:
    resolved_6f = resolve_model(
        latent_variables=[lv_6f],
        likert_indicators=indicators_6f,
        likert_types=types_6f,
        measurement_configuration=ordinal_measurement_configuration(indicators_6f),
        context=BuildContext.default(EstimationMode.MAXIMUM_LIKELIHOOD),
        normalization_plan=NormalizationPlan(),
    )
    print(format_resolved_summary(resolved_6f))
except Exception as e:
    print(f'Expected problem detected: {e}')
