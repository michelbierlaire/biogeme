"""
Generate files from latent-variable measurement specifications
===============================================================

Generate files from latent-variable measurement specifications.

This script illustrates the latent-variables specification API. It does not
estimate any model and does not use data. Instead, it starts from pure
specification objects, resolves them into semantic ``ResolvedModel`` objects, and
uses the resolved specifications to generate files that can be inspected or used
elsewhere.

The script shows how to:

- define pure latent-variable and measurement specifications,
- define normalization plans in expert mode, using explicit parameter fixings,
- resolve a specification into a semantic ``ResolvedModel``,
- build live Biogeme expressions from the resolved specification,
- generate files from the resolved specification:
   - pedagogical runnable Python code,
   - a LaTeX scientific report,
   - a static HTML report.

The same specification is resolved under several build contexts:

- maximum likelihood and Bayesian estimation modes,
- log/exp and lower-bound parameterizations for positive parameters,
- automatic normalization and an explicit normalization plan.

Only specification files and reports are generated. No estimation algorithm is
run in this example.

Michel Bierlaire
Sat Jun 06 2026, 15:20:15
"""

from __future__ import annotations

from pathlib import Path

from likert_spec import likert_indicators, likert_types
from two_latent_variables_spec import latent_variables

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
    generate_html_report,
    generate_latex_report,
    generate_python_code,
    resolve_model,
    save_html_report,
    save_latex_report,
    save_python_code,
)

OUTPUT_DIR = Path('b01_generated_files')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('\n====================')
print('SPEC loaded')
print('====================')

print(f'- #latent_variables:  {len(latent_variables)}')
print(f'- #likert_indicators: {len(likert_indicators)}')
print(f'- #likert_types:      {len(likert_types)}')

# =============================================================================
# Homogeneous ordinal measurement configuration used by the resolver
# =============================================================================
DEFAULT_SIGMA_START = 10.0

measurement_configuration = MeasurementConfiguration(
    specifications=[
        IndicatorMeasurementSpec(
            indicator_name=indicator.name,
            measurement_model=MeasurementModel.ORDERED_PROBIT,
            measurement_sigma=PositiveParameterSpec(start=DEFAULT_SIGMA_START),
        )
        for indicator in likert_indicators
    ]
)

# =============================================================================
# Normalization plans used when resolving the specifications
# =============================================================================
# Plan A: empty plan. The resolver determines the required normalizations.
plan_empty = NormalizationPlan()

# Plan B: explicit plan. These fixings illustrate how expert-specified
# normalizations are passed to the resolver.
plan_example = NormalizationPlan()

# --- LV normalizations by reference-indicator strategy (examples)
# car_centric_attitude anchored on Envir01
plan_example.add(
    Fixing(MeasurementIntercept('Envir01'), 0.0, note='reference indicator (location)')
)
plan_example.add(
    Fixing(
        MeasurementLoading('car_centric_attitude', 'Envir01'),
        -1.0,
        note='reference indicator (scale)',
    )
)

# environmental_attitude anchored on Envir02
plan_example.add(
    Fixing(MeasurementIntercept('Envir02'), 0.0, note='reference indicator (location)')
)
plan_example.add(
    Fixing(
        MeasurementLoading('environmental_attitude', 'Envir02'),
        1.0,
        note='reference indicator (scale)',
    )
)

# --- Ordinal scale normalization (one per ordinal threshold system)
# With a single ordinal threshold system ("likert"), fix one measurement sigma.
plan_example.add(
    Fixing(MeasurementSigma('Envir01'), 1.0, note='ordinal threshold scale')
)

print('\n====================')
print('Normalization plans ready')
print('====================')
print('- plan_empty:   (no fixings)')
print('- plan_example: (some intercept/loadings/sigma fixed)')


# =============================================================================
# Helpers
# =============================================================================
def make_context(
    estimation_mode: EstimationMode,
    positivity_mode: PositivityMode,
) -> BuildContext:
    """Create a build context used to resolve and generate a specification."""
    base = BuildContext.default(estimation_mode)
    return BuildContext(
        estimation_mode=base.estimation_mode,
        draw_type=base.draw_type,
        positivity_mode=positivity_mode,
        naming=base.naming,
        ordinal_eps=base.ordinal_eps,
        ordinal_enforce_order=base.ordinal_enforce_order,
    )


def title_to_file_stem(title: str) -> str:
    """Convert a case title into a filesystem-friendly stem."""
    stem = title.lower()
    replacements = {
        ' ': '_',
        '/': '_',
        '+': 'plus',
        ':': '',
        ',': '',
        '(': '',
        ')': '',
        '-': '_',
    }
    for source, target in replacements.items():
        stem = stem.replace(source, target)
    while '__' in stem:
        stem = stem.replace('__', '_')
    return stem.strip('_')


def summarize_case(
    *, title: str, context: BuildContext, plan: NormalizationPlan
) -> None:
    """Resolve one specification, generate files, and print a compact summary.

    No estimation is performed. The generated Python, LaTeX, and HTML files are
    derived from the resolved specification.
    """
    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)

    resolved = resolve_model(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        measurement_configuration=measurement_configuration,
        context=context,
        normalization_plan=plan,
    )
    built = build_biogeme_model(resolved)
    python_code = generate_python_code(resolved)
    latex_report = generate_latex_report(resolved)
    html_report = generate_html_report(resolved)

    stem = title_to_file_stem(title)
    python_path = OUTPUT_DIR / f'{stem}.py'
    latex_path = OUTPUT_DIR / f'{stem}.tex'
    html_path = OUTPUT_DIR / f'{stem}.html'

    save_python_code(python_code, python_path)
    save_latex_report(latex_report, latex_path)
    save_html_report(html_report, html_path)

    measurement_models = [
        model.value for model in resolved.metadata.measurement_models_present
    ]

    print(f'- estimation mode: {context.estimation_mode.value}')
    print(f'- positivity mode: {context.positivity_mode.value}')
    print(f'- latent variables: {resolved.metadata.n_latent_variables}')
    print(f'- indicators: {resolved.metadata.n_indicators}')
    print(f'- ordinal threshold systems: {resolved.metadata.n_threshold_systems}')
    print(f'- measurement models present: {measurement_models}')
    print(f'- resolved parameters: {len(resolved.parameters)}')
    print(f'- structural expressions: {list(built.latent_expressions.keys())}')
    print(f'- threshold systems: {list(built.threshold_expressions.keys())}')
    print(f'- measurement terms: {len(built.measurement_terms)} indicators')
    print(f'  example term (Envir01): {built.measurement_terms.get("Envir01")}')
    print(f'- generated Python file: {python_path.as_posix()}')
    print(f'- generated LaTeX file: {latex_path.as_posix()}')
    print(f'- generated HTML file: {html_path.as_posix()}')

    if resolved.normalization.rules:
        print('- normalization rules:')
        for rule in resolved.normalization.rules:
            print(f'  * {rule.target_name} = {rule.value} ({rule.reason})')
    else:
        print('- normalization rules: none')

    if resolved.normalization.warnings:
        print('- warnings:')
        for warning in resolved.normalization.warnings:
            print(f'  * {warning}')
    else:
        print('- warnings: none')


# =============================================================================
# Enumerate build contexts and normalization plans explicitly
# =============================================================================
# We generate files for 8 cases:
#   modes: maximum likelihood vs Bayesian,
#   positivity: log/exp vs lower-bound parameterization,
#   normalization plan: automatic vs explicit.
#
# Some combinations are not intended as recommended estimation settings. They
# are included to exercise the specification resolver and code generators.

# -------------------------------------------------------------------------
# CASE 1: ML + log/exp positivity + empty plan
# -------------------------------------------------------------------------
context_1 = make_context(
    EstimationMode.MAXIMUM_LIKELIHOOD,
    PositivityMode.LOG_EXP,
)
summarize_case(
    title='CASE 1: ML + log/exp positivity + EMPTY plan',
    context=context_1,
    plan=plan_empty,
)

# -------------------------------------------------------------------------
# CASE 2: ML + log/exp positivity + example plan
# -------------------------------------------------------------------------
context_2 = make_context(
    EstimationMode.MAXIMUM_LIKELIHOOD,
    PositivityMode.LOG_EXP,
)
summarize_case(
    title='CASE 2: ML + log/exp positivity + EXAMPLE plan',
    context=context_2,
    plan=plan_example,
)

# -------------------------------------------------------------------------
# CASE 3: ML + bounded positivity + empty plan
# -------------------------------------------------------------------------
context_3 = make_context(
    EstimationMode.MAXIMUM_LIKELIHOOD,
    PositivityMode.LOWER_BOUND,
)
summarize_case(
    title='CASE 3: ML + BOUNDED positivity + EMPTY plan (non-standard but illustrative)',
    context=context_3,
    plan=plan_empty,
)

# -------------------------------------------------------------------------
# CASE 4: ML + bounded positivity + example plan
# -------------------------------------------------------------------------
context_4 = make_context(
    EstimationMode.MAXIMUM_LIKELIHOOD,
    PositivityMode.LOWER_BOUND,
)
summarize_case(
    title='CASE 4: ML + BOUNDED positivity + EXAMPLE plan (non-standard but illustrative)',
    context=context_4,
    plan=plan_example,
)

# -------------------------------------------------------------------------
# CASE 5: Bayesian + bounded positivity + empty plan
# -------------------------------------------------------------------------
context_5 = make_context(
    EstimationMode.BAYESIAN,
    PositivityMode.LOWER_BOUND,
)
summarize_case(
    title='CASE 5: BAYESIAN + BOUNDED positivity + EMPTY plan',
    context=context_5,
    plan=plan_empty,
)

# -------------------------------------------------------------------------
# CASE 6: Bayesian + bounded positivity + example plan
# -------------------------------------------------------------------------
context_6 = make_context(
    EstimationMode.BAYESIAN,
    PositivityMode.LOWER_BOUND,
)
summarize_case(
    title='CASE 6: BAYESIAN + BOUNDED positivity + EXAMPLE plan',
    context=context_6,
    plan=plan_example,
)

# -------------------------------------------------------------------------
# CASE 7: Bayesian + log/exp positivity + empty plan
# -------------------------------------------------------------------------
context_7 = make_context(
    EstimationMode.BAYESIAN,
    PositivityMode.LOG_EXP,
)
summarize_case(
    title='CASE 7: BAYESIAN + log/exp positivity + EMPTY plan (non-standard but illustrative)',
    context=context_7,
    plan=plan_empty,
)

# -------------------------------------------------------------------------
# CASE 8: Bayesian + log/exp positivity + example plan
# -------------------------------------------------------------------------
context_8 = make_context(
    EstimationMode.BAYESIAN,
    PositivityMode.LOG_EXP,
)
summarize_case(
    title='CASE 8: BAYESIAN + log/exp positivity + EXAMPLE plan (non-standard but illustrative)',
    context=context_8,
    plan=plan_example,
)

print('\n====================')
print('END OF SCRIPT')
print('====================')
