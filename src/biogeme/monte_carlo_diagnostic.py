"""Post-estimation Monte Carlo draw-stability diagnostics.

The diagnostic evaluates a model criterion and its gradient at a fixed
estimated parameter vector using fresh draw designs.  It never optimizes and
constructs its JAX evaluators with second derivatives disabled.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import secrets
import signal
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import yaml

from biogeme.draws import DrawsManagement
from biogeme.draws.factory import DrawFactory
from biogeme.draws.generators import get_halton_draws, get_normal_wichura_draws
from biogeme.draws.native_draws import native_random_number_generators
from biogeme.exceptions import BiogemeError
from biogeme.floating_point import JAX_FLOAT
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.second_derivatives import SecondDerivativesMode

if TYPE_CHECKING:
    from biogeme.model_elements import ModelElements
    from biogeme.parameters import Parameters
    from biogeme.results_processing import EstimationResults

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MINIMUM_DRAW_COUNT = 2
HALTON_PATTERN = re.compile(r'^(UNIFORM|UNIFORMSYM|NORMAL)_HALTON([235])$')

EXECUTION_STATUSES = {
    'not_started',
    'running',
    'completed',
    'interrupted',
    'time_budget_exceeded',
    'failed',
}
DIAGNOSTIC_CONCLUSIONS = {
    'stable',
    'unstable',
    'inconclusive',
    'not_applicable',
}
RECOMMENDATIONS = {
    'no_more_draws_indicated',
    'more_draws_recommended',
    'additional_diagnostics_needed',
    'not_applicable',
}


def utc_now() -> str:
    """Return a timezone-aware timestamp suitable for YAML output."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


@dataclass(frozen=True)
class MonteCarloDiagnosticConfiguration:
    """Validated configuration for a diagnostic run."""

    draw_factors: tuple[float, ...]
    replications: int
    time_budget_seconds: float
    max_draws: int
    safety_factor: float
    objective_tolerance: float
    gradient_tolerance: float
    minimum_level_factor: float

    @classmethod
    def from_parameters(
        cls, parameters: Parameters
    ) -> MonteCarloDiagnosticConfiguration:
        """Read and validate diagnostic settings from Biogeme parameters."""
        raw_factors = parameters.get_value('monte_carlo_diagnostic_draw_factors')
        if not isinstance(raw_factors, str):
            raise BiogemeError(
                'monte_carlo_diagnostic_draw_factors must be a comma-separated string.'
            )
        try:
            draw_factors = tuple(
                float(item.strip()) for item in raw_factors.split(',') if item.strip()
            )
        except ValueError as error:
            raise BiogemeError(
                'monte_carlo_diagnostic_draw_factors contains a nonnumeric value: '
                f'{raw_factors!r}.'
            ) from error
        if not draw_factors or any(
            not np.isfinite(factor) or factor <= 0 for factor in draw_factors
        ):
            raise BiogemeError(
                'monte_carlo_diagnostic_draw_factors must contain only positive, '
                'finite values.'
            )

        configuration = cls(
            draw_factors=draw_factors,
            replications=int(
                parameters.get_value('monte_carlo_diagnostic_replications')
            ),
            time_budget_seconds=float(
                parameters.get_value('monte_carlo_diagnostic_time_budget')
            ),
            max_draws=int(parameters.get_value('monte_carlo_diagnostic_max_draws')),
            safety_factor=float(
                parameters.get_value('monte_carlo_diagnostic_safety_factor')
            ),
            objective_tolerance=float(
                parameters.get_value('monte_carlo_diagnostic_objective_tolerance')
            ),
            gradient_tolerance=float(
                parameters.get_value('monte_carlo_diagnostic_gradient_tolerance')
            ),
            minimum_level_factor=float(
                parameters.get_value('monte_carlo_diagnostic_minimum_level_factor')
            ),
        )
        configuration.validate()
        return configuration

    def validate(self) -> None:
        """Validate relationships not covered by the scalar parameter system."""
        finite_values = {
            'time budget': self.time_budget_seconds,
            'safety factor': self.safety_factor,
            'objective tolerance': self.objective_tolerance,
            'gradient tolerance': self.gradient_tolerance,
            'minimum level factor': self.minimum_level_factor,
        }
        nonfinite = [
            name for name, value in finite_values.items() if not np.isfinite(value)
        ]
        if nonfinite:
            raise BiogemeError(
                'Monte Carlo diagnostic configuration values must be finite: '
                f'{", ".join(nonfinite)}.'
            )
        if self.replications <= 0:
            raise BiogemeError('Diagnostic replications must be positive.')
        if self.time_budget_seconds <= 0:
            raise BiogemeError('The diagnostic time budget must be positive.')
        if self.max_draws < MINIMUM_DRAW_COUNT:
            raise BiogemeError(
                f'The diagnostic maximum draw count must be at least '
                f'{MINIMUM_DRAW_COUNT}.'
            )
        if self.safety_factor < 1.0:
            raise BiogemeError(
                'The Monte Carlo diagnostic safety factor must be at least 1.'
            )
        if self.objective_tolerance < 0 or self.gradient_tolerance < 0:
            raise BiogemeError('Diagnostic tolerances cannot be negative.')
        if self.minimum_level_factor <= 1.0:
            raise BiogemeError(
                'monte_carlo_diagnostic_minimum_level_factor must be greater '
                'than 1 so that evidence above the estimation draw count is required.'
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a YAML-safe representation."""
        return {
            'draw_factors': list(self.draw_factors),
            'replications': self.replications,
            'time_budget_seconds': self.time_budget_seconds,
            'max_draws': self.max_draws,
            'safety_factor': self.safety_factor,
            'objective_tolerance': self.objective_tolerance,
            'gradient_tolerance': self.gradient_tolerance,
            'minimum_level_factor': self.minimum_level_factor,
        }


@dataclass(frozen=True)
class MonteCarloDiagnosticResult:
    """Result returned by :meth:`BIOGEME.check_monte_carlo_stability`."""

    data: dict[str, Any]
    yaml_file: Path
    markdown_file: Path

    @property
    def execution_status(self) -> str:
        return str(self.data['execution_status'])

    @property
    def diagnostic_conclusion(self) -> str:
        return str(self.data['diagnostic_conclusion'])

    @property
    def recommendation(self) -> str:
        return str(self.data['recommendation'])


def has_antithetic_draws(draw_types: dict[str, str]) -> bool:
    """Return whether any requested draw type requires complete pairs."""
    return any('_ANTI' in draw_type.upper() for draw_type in draw_types.values())


def normalize_draw_count(
    requested: float,
    max_draws: int,
    antithetic: bool,
) -> int | None:
    """Normalize a requested count while preserving antithetic pairs."""
    count = max(MINIMUM_DRAW_COUNT, int(round(requested)))
    if antithetic and count % 2:
        count += 1
    if count > max_draws:
        return None
    return count


def build_draw_schedule(
    original_draws: int,
    configuration: MonteCarloDiagnosticConfiguration,
    draw_types: dict[str, str],
) -> list[int]:
    """Build a unique, fast-first draw schedule."""
    if original_draws <= 0:
        raise BiogemeError(
            f'The original Monte Carlo draw count must be positive, not '
            f'{original_draws}.'
        )
    antithetic = has_antithetic_draws(draw_types)
    counts = {
        count
        for factor in configuration.draw_factors
        if (
            count := normalize_draw_count(
                requested=original_draws * factor,
                max_draws=configuration.max_draws,
                antithetic=antithetic,
            )
        )
        is not None
    }
    if not counts:
        raise BiogemeError(
            'The configured factors and maximum draw count produce no valid '
            'diagnostic draw levels.'
        )
    return sorted(counts)


def diagnostic_task_seed(base_seed: int, draw_count: int, replication: int) -> int:
    """Derive a deterministic, task-specific 32-bit seed."""
    sequence = np.random.SeedSequence([base_seed, draw_count, replication])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def build_tasks(
    schedule: list[int],
    replications: int,
    original_draws: int,
    draw_types: dict[str, str],
    base_seed: int,
) -> list[dict[str, Any]]:
    """Build deterministic task records sorted by expected runtime."""
    return [
        {
            'draw_count': draw_count,
            'draw_factor': draw_count / original_draws,
            'replication': replication,
            'draw_types': dict(draw_types),
            'seed': diagnostic_task_seed(base_seed, draw_count, replication),
            'randomization_identifier': (
                f'diagnostic-seed-{diagnostic_task_seed(base_seed, draw_count, replication)}'
            ),
        }
        for draw_count in schedule
        for replication in range(1, replications + 1)
    ]


def _randomized_halton(
    draw_type: str,
    sample_size: int,
    number_of_draws: int,
) -> np.ndarray:
    """Generate a diagnostic-only randomized native Halton design."""
    match = HALTON_PATTERN.fullmatch(draw_type.upper())
    if match is None:
        raise BiogemeError(f'Not a supported native Halton type: {draw_type}.')
    family, base_text = match.groups()
    uniform = get_halton_draws(
        sample_size=sample_size,
        number_of_draws=number_of_draws,
        symmetric=False,
        base=int(base_text),
        skip=10,
    )
    # A Cranley-Patterson modulo-one shift leaves the ordinary Halton generator
    # unchanged while providing independent randomized diagnostic replications.
    shifts = np.random.uniform(size=(sample_size, 1))
    shifted = np.mod(uniform + shifts, 1.0)
    if family == 'UNIFORM':
        return shifted
    if family == 'UNIFORMSYM':
        return 2.0 * shifted - 1.0
    epsilon = np.finfo(float).eps
    shifted = np.clip(shifted, epsilon, 1.0 - epsilon)
    return get_normal_wichura_draws(
        sample_size=sample_size,
        number_of_draws=number_of_draws,
        uniform_numbers=shifted.copy(),
    )


def _validate_antithetic_pairs(draw_type: str, draws: np.ndarray) -> None:
    """Verify native antithetic designs have complete, matching pairs."""
    normalized = draw_type.upper()
    if '_ANTI' not in normalized or normalized not in native_random_number_generators:
        return
    if draws.shape[1] % 2:
        raise BiogemeError(
            f'Antithetic draw type {draw_type} produced an odd number of columns.'
        )
    half = draws.shape[1] // 2
    first = draws[:, :half]
    second = draws[:, half:]
    expected = 1.0 - first if normalized.startswith('UNIFORM_') else -first
    if not np.allclose(second, expected):
        raise BiogemeError(
            f'Antithetic draw type {draw_type} did not produce complete pairs.'
        )


def generate_diagnostic_draws(
    draw_types: dict[str, str],
    variable_names: list[str],
    sample_size: int,
    number_of_draws: int,
    seed: int,
    user_generators: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, str], list[str]]:
    """Generate one fresh, reproducible diagnostic draw design.

    The global NumPy RNG state is restored afterward because Biogeme's native
    generators currently use that RNG.  Ordinary estimation draw behavior is
    therefore unaffected by diagnostic generation.
    """
    if number_of_draws <= 0:
        raise BiogemeError('The diagnostic draw count must be positive.')
    if has_antithetic_draws(draw_types) and number_of_draws % 2:
        raise BiogemeError(
            'Antithetic diagnostic draw counts must be even to preserve pairs.'
        )
    factory = DrawFactory(user_generators)
    specs = factory.make_draw_specs(draw_types, variable_names)
    arrays: list[np.ndarray] = []
    methods: dict[str, str] = {}
    limitations: list[str] = []
    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        for spec in specs:
            normalized = spec.draw_type.upper()
            if HALTON_PATTERN.fullmatch(normalized):
                array = _randomized_halton(
                    draw_type=normalized,
                    sample_size=sample_size,
                    number_of_draws=number_of_draws,
                )
                methods[spec.name] = 'randomized_halton_modulo_one_shift'
            else:
                array = spec.generator(sample_size, number_of_draws)
                methods[spec.name] = (
                    'fresh_native_design'
                    if normalized in native_random_number_generators
                    else 'fresh_user_defined_design'
                )
                if normalized not in native_random_number_generators:
                    limitations.append(
                        f'Independence of user-defined draw type {spec.draw_type!r} '
                        'depends on its generator honoring NumPy seeding.'
                    )
            if array.shape != (sample_size, number_of_draws):
                raise BiogemeError(
                    f'Diagnostic draws for {spec.name!r} have shape {array.shape}; '
                    f'expected {(sample_size, number_of_draws)}.'
                )
            _validate_antithetic_pairs(spec.draw_type, array)
            arrays.append(array)
    finally:
        np.random.set_state(previous_state)
    return np.moveaxis(np.asarray(arrays), 0, -1), methods, sorted(set(limitations))


def task_identity(task: dict[str, Any]) -> tuple[Any, ...]:
    """Return the persistent identity of a planned or completed task."""
    draw_types = tuple(sorted(dict(task['draw_types']).items()))
    return (
        int(task['draw_count']),
        int(task['replication']),
        draw_types,
        int(task['seed']),
        str(task['randomization_identifier']),
    )


def forecast_remaining_seconds(
    completed: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    safety_factor: float,
) -> float | None:
    """Forecast pending runtime using median seconds per effective draw."""
    normalized_durations = [
        float(item['elapsed_seconds']) / int(item['draw_count'])
        for item in completed
        if float(item.get('elapsed_seconds', 0.0)) >= 0 and int(item['draw_count']) > 0
    ]
    if not normalized_durations:
        return None
    seconds_per_draw = median(normalized_durations)
    return float(
        safety_factor
        * seconds_per_draw
        * sum(int(task['draw_count']) for task in pending)
    )


def diagnostic_conclusion(
    completed: list[dict[str, Any]],
    original_draws: int,
    configuration: MonteCarloDiagnosticConfiguration,
) -> tuple[str, str, list[str]]:
    """Calculate conclusion, recommendation, and machine-readable motivation."""
    if not completed:
        return (
            'inconclusive',
            'additional_diagnostics_needed',
            ['no_completed_evaluations'],
        )

    def is_stable(item: dict[str, Any]) -> bool:
        return (
            float(item['objective_difference']) <= configuration.objective_tolerance
            and float(item['gradient_linf_difference'])
            <= configuration.gradient_tolerance
        )

    at_original = [
        item for item in completed if int(item['draw_count']) == original_draws
    ]
    minimum_high_draws = original_draws * configuration.minimum_level_factor
    high = [item for item in completed if int(item['draw_count']) >= minimum_high_draws]
    highest: list[dict[str, Any]] = []
    if high:
        highest_draw_count = max(int(item['draw_count']) for item in high)
        highest = [
            item for item in high if int(item['draw_count']) == highest_draw_count
        ]
    evidence = at_original + highest
    if at_original and highest and all(is_stable(item) for item in evidence):
        return (
            'stable',
            'no_more_draws_indicated',
            [
                'objective_stable_at_original_draw_count',
                'gradient_stable_at_original_draw_count',
                'objective_stable_at_higher_draw_count',
                'gradient_stable_at_higher_draw_count',
            ],
        )

    if highest:
        if any(not is_stable(item) for item in highest):
            codes = ['discrepancy_above_tolerance_at_higher_draw_count']
            if any(
                float(item['objective_difference']) > configuration.objective_tolerance
                for item in highest
            ):
                codes.append('objective_unstable_at_higher_draw_count')
            if any(
                float(item['gradient_linf_difference'])
                > configuration.gradient_tolerance
                for item in highest
            ):
                codes.append('gradient_unstable_at_higher_draw_count')
            return 'unstable', 'more_draws_recommended', codes

    return (
        'inconclusive',
        'additional_diagnostics_needed',
        ['minimum_high_draw_evidence_not_completed'],
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a structured diagnostic checkpoint atomically."""
    atomic_write_text(
        path,
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )


def _humanized(value: str) -> str:
    return value.replace('_', ' ').capitalize()


def _motivation_text(codes: list[str]) -> str:
    descriptions = {
        'no_completed_evaluations': 'No diagnostic evaluation has completed.',
        'minimum_high_draw_evidence_not_completed': (
            'The diagnostic did not complete both the original draw level and '
            'the configured higher-draw evidence level.'
        ),
        'objective_stable_at_original_draw_count': (
            'The objective discrepancy at the original draw count is within tolerance.'
        ),
        'gradient_stable_at_original_draw_count': (
            'The gradient discrepancy at the original draw count is within tolerance.'
        ),
        'objective_stable_at_higher_draw_count': (
            'The objective discrepancy at a higher draw count is within tolerance.'
        ),
        'gradient_stable_at_higher_draw_count': (
            'The gradient discrepancy at a higher draw count is within tolerance.'
        ),
        'discrepancy_above_tolerance_at_higher_draw_count': (
            'At least one discrepancy remains above tolerance at the highest '
            'completed draw level.'
        ),
        'objective_unstable_at_higher_draw_count': (
            'The objective discrepancy remains above tolerance at the highest '
            'completed draw level.'
        ),
        'gradient_unstable_at_higher_draw_count': (
            'The gradient discrepancy remains above tolerance at the highest '
            'completed draw level.'
        ),
        'model_has_no_monte_carlo_integration': (
            'The estimated model does not contain a Monte Carlo expression.'
        ),
    }
    return '\n'.join(f'- {descriptions.get(code, _humanized(code))}' for code in codes)


def generate_markdown_report(data: dict[str, Any]) -> str:
    """Generate the separate American-English diagnostic report."""
    status = str(data['execution_status'])
    conclusion = str(data['diagnostic_conclusion'])
    recommendation = str(data['recommendation'])
    completed = list(data.get('completed_evaluations', []))
    skipped = list(data.get('skipped_evaluations', []))
    completed_ids = {task_identity(item) for item in completed}
    uncompleted = [
        item
        for item in data.get('planned_evaluations', [])
        if task_identity(item) not in completed_ids
    ]
    skipped_ids = {task_identity(item) for item in skipped}
    planned_not_started = [
        item for item in uncompleted if task_identity(item) not in skipped_ids
    ]

    if status == 'interrupted':
        status_detail = (
            f'Interrupted by the user after completing {len(completed)} evaluation(s).'
        )
    elif status == 'time_budget_exceeded':
        status_detail = (
            'The projected runtime exceeded the configured time budget before '
            'the next evaluation started.'
        )
    elif status == 'completed':
        status_detail = 'All planned evaluations completed.'
    elif status == 'failed':
        status_detail = f'The diagnostic failed: {data.get("failure", "unknown error")}'
    else:
        status_detail = 'The diagnostic is in progress.'

    recommendation_text = {
        'no_more_draws_indicated': (
            'No additional draws appear necessary. The completed high-draw '
            'objective and gradient differences are within the configured tolerances.'
        ),
        'more_draws_recommended': (
            'More draws are recommended because discrepancies remain above '
            'the configured tolerances at a higher draw level.'
        ),
        'additional_diagnostics_needed': (
            'Additional diagnostics are needed because the required higher-draw '
            'evidence was not completed.'
        ),
        'not_applicable': (
            'This diagnostic is not applicable because the model contains no '
            'Monte Carlo integration.'
        ),
    }[recommendation]

    configuration = data['configuration']
    rows = []
    for item in completed:
        rows.append(
            '| {draw_count} | {replication} | {objective:.10g} | '
            '{objective_difference:.4g} | {relative_objective_difference:.4g} | '
            '{gradient_linf_difference:.4g} | {gradient_l2_difference:.4g} | '
            '{elapsed_seconds:.3f} |'.format(**item)
        )
    results_table = (
        '\n'.join(
            [
                '| Draws | Replication | Objective | Absolute objective difference | '
                'Relative objective difference | Gradient $L_\\infty$ difference | '
                'Gradient $L_2$ difference | Seconds |',
                '|---:|---:|---:|---:|---:|---:|---:|---:|',
                *rows,
            ]
        )
        if rows
        else '_No evaluation has completed._'
    )

    not_started_lines = (
        '\n'.join(
            f'- {item["draw_count"]} draws, replication {item["replication"]}'
            for item in planned_not_started
        )
        or '- None.'
    )
    skipped_lines = (
        '\n'.join(
            f'- {item["draw_count"]} draws, replication {item["replication"]}: '
            f'{_humanized(str(item.get("reason", "not started")))}'
            for item in skipped
        )
        or '- None.'
    )
    seed_lines = (
        '\n'.join(
            f'- {item["draw_count"]} draws, replication {item["replication"]}: '
            f'seed {item["seed"]} ({item["randomization_identifier"]})'
            for item in data.get('planned_evaluations', [])
        )
        or '- None.'
    )
    limitation_lines = (
        '\n'.join(f'- {item}' for item in data.get('limitations', []))
        or '- No draw-design limitation was identified.'
    )
    interruption_details = (
        f'- Completed evaluations: {len(completed)}\n'
        f'- Uncompleted evaluations: {len(uncompleted)}\n'
        f'- Skipped because of the time budget: {len(skipped)}'
    )

    return f"""# Monte Carlo Draw-Stability Diagnostic: {data['model_name']}

## Execution status

**Execution status:** {_humanized(status)}.

{status_detail}

## Diagnostic conclusion

**Diagnostic conclusion:** {_humanized(conclusion)}.

## Recommendation

**Recommendation:** {_humanized(recommendation)}.

{recommendation_text}

## Motivation for the recommendation

{_motivation_text(list(data.get('motivation_codes', [])))}

## Purpose and limitations

This practical diagnostic assesses sensitivity to the number and design of
Monte Carlo draws. It is not a formal integration-error confidence interval.
No re-estimation was performed. The estimated parameter vector remained fixed.

{limitation_lines}

## Methodology

The original estimation result is the reference. For each planned draw level,
the model criterion and its gradient were evaluated at the fixed estimated
parameters using a fresh diagnostic draw design. Pseudo-random, antithetic,
and MLHS designs were regenerated independently. Native Halton designs used a
diagnostic-only randomized modulo-one shift; ordinary estimation behavior was
not changed. No Hessian was requested. The first Ctrl-C requests a graceful
stop after the active evaluation has been checkpointed; a second Ctrl-C may
terminate immediately.

## Explanation of calculated quantities

The objective is the model criterion evaluated at the fixed estimated
parameters. The gradient is the derivative of that criterion at those same
parameters. Objective discrepancies measure sensitivity of the criterion to
the draw design. Gradient discrepancies are especially important because they
indicate whether the simulated optimum may move. The infinity norm is the
largest absolute component of the gradient difference; the Euclidean norm
summarizes its overall magnitude.

## Configuration

- Original number of draws: {data['original_number_of_draws']}
- Draw factors: {configuration['draw_factors']}
- Replications per level: {configuration['replications']}
- Time budget: {configuration['time_budget_seconds']} seconds
- Maximum draws: {configuration['max_draws']}
- Runtime safety factor: {configuration['safety_factor']}
- Objective tolerance: {configuration['objective_tolerance']}
- Gradient tolerance: {configuration['gradient_tolerance']}
- Minimum conclusive level factor: {configuration['minimum_level_factor']}

## Results

{results_table}

## Interruption or time-budget details

{interruption_details}

## Completed and uncompleted evaluations

Completed: {len(completed)} of {len(data.get('planned_evaluations', []))}.

Planned but not started:

{not_started_lines}

Skipped because of the time budget:

{skipped_lines}

## Seeds and randomization identifiers

{seed_lines}

## Suggested next action

{recommendation_text}
"""


def _validate_resumed_data(
    data: dict[str, Any],
    baseline: dict[str, Any],
    planned_tasks: list[dict[str, Any]],
) -> None:
    """Reject a checkpoint that belongs to another model or task plan."""
    if data.get('schema_version') != SCHEMA_VERSION:
        raise BiogemeError('Unsupported Monte Carlo diagnostic schema version.')
    comparable_fields = (
        'model_name',
        'original_number_of_draws',
        'estimated_parameters',
        'original_result',
        'draw_types',
    )
    for field in comparable_fields:
        if data.get(field) != baseline.get(field):
            raise BiogemeError(
                f'Existing Monte Carlo diagnostic checkpoint has incompatible {field}.'
            )
    existing_tasks = [task_identity(item) for item in data['planned_evaluations']]
    new_tasks = [task_identity(item) for item in planned_tasks]
    if existing_tasks != new_tasks:
        raise BiogemeError(
            'Existing Monte Carlo diagnostic checkpoint has a different draw '
            'schedule or replication plan.'
        )


class MonteCarloDiagnosticRunner:
    """Checkpointed, interruptible execution of diagnostic tasks."""

    def __init__(
        self,
        baseline: dict[str, Any],
        configuration: MonteCarloDiagnosticConfiguration,
        planned_tasks: list[dict[str, Any]],
        evaluate_task: Callable[[dict[str, Any]], dict[str, Any]],
        yaml_file: Path,
        markdown_file: Path,
        base_seed: int,
        limitations: list[str] | None = None,
        resume: bool = True,
    ):
        self.baseline = baseline
        self.configuration = configuration
        self.planned_tasks = planned_tasks
        self.evaluate_task = evaluate_task
        self.yaml_file = yaml_file
        self.markdown_file = markdown_file
        self.base_seed = base_seed
        self.limitations = list(limitations or [])
        self.resume = resume
        self.stop_requested = False
        self.interrupt_count = 0

    def request_stop(self) -> None:
        """Request a graceful stop after the current evaluation."""
        self.stop_requested = True

    def _signal_handler(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.interrupt_count += 1
        if self.interrupt_count == 1:
            logger.warning(
                'Monte Carlo diagnostic interruption requested. The current '
                'evaluation will be checkpointed when it finishes.'
            )
            self.request_stop()
            return
        raise KeyboardInterrupt

    def _new_data(self) -> dict[str, Any]:
        now = utc_now()
        return {
            'schema_version': SCHEMA_VERSION,
            **self.baseline,
            'created_at': now,
            'updated_at': now,
            'execution_status': 'not_started',
            'diagnostic_conclusion': 'inconclusive',
            'recommendation': 'additional_diagnostics_needed',
            'configuration': self.configuration.as_dict(),
            'base_seed': self.base_seed,
            'planned_evaluations': self.planned_tasks,
            'completed_evaluations': [],
            'skipped_evaluations': [],
            'timing': {
                'elapsed_seconds': 0.0,
                'projected_remaining_seconds': None,
            },
            'motivation_codes': ['no_completed_evaluations'],
            'limitations': list(self.limitations),
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        if self.resume and self.yaml_file.exists():
            with self.yaml_file.open('r', encoding='utf-8') as stream:
                loaded = yaml.safe_load(stream)
            if not isinstance(loaded, dict):
                raise BiogemeError(
                    f'Invalid Monte Carlo diagnostic checkpoint: {self.yaml_file}.'
                )
            _validate_resumed_data(loaded, self.baseline, self.planned_tasks)
            loaded['configuration'] = self.configuration.as_dict()
            loaded['skipped_evaluations'] = []
            loaded['failure'] = None
            loaded.setdefault('limitations', [])
            loaded['limitations'] = sorted(
                set(loaded['limitations']) | set(self.limitations)
            )
            return loaded
        return self._new_data()

    def _update_conclusion(self, data: dict[str, Any]) -> None:
        conclusion, recommendation, codes = diagnostic_conclusion(
            completed=list(data['completed_evaluations']),
            original_draws=int(data['original_number_of_draws']),
            configuration=self.configuration,
        )
        data['diagnostic_conclusion'] = conclusion
        data['recommendation'] = recommendation
        data['motivation_codes'] = codes

    def _checkpoint(self, data: dict[str, Any]) -> None:
        self._update_conclusion(data)
        data['updated_at'] = utc_now()
        atomic_write_yaml(self.yaml_file, data)
        atomic_write_text(self.markdown_file, generate_markdown_report(data))

    def run(self) -> MonteCarloDiagnosticResult:
        """Run or resume the diagnostic and return its checkpointed result."""
        data = self._load_or_initialize()
        completed_ids = {task_identity(item) for item in data['completed_evaluations']}
        pending = [
            task
            for task in self.planned_tasks
            if task_identity(task) not in completed_ids
        ]
        old_handler: Any = None
        handler_installed = False
        try:
            try:
                old_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._signal_handler)
                handler_installed = True
            except ValueError:
                # Signal handlers can only be installed in the main thread.
                logger.info(
                    'SIGINT handler not installed because the diagnostic is not '
                    'running in the main thread.'
                )

            data['execution_status'] = 'running'
            self._checkpoint(data)

            while pending:
                completed = list(data['completed_evaluations'])
                elapsed = sum(float(item['elapsed_seconds']) for item in completed)
                forecast = forecast_remaining_seconds(
                    completed=completed,
                    pending=pending,
                    safety_factor=self.configuration.safety_factor,
                )
                data['timing']['elapsed_seconds'] = elapsed
                data['timing']['projected_remaining_seconds'] = forecast
                if elapsed >= self.configuration.time_budget_seconds or (
                    forecast is not None
                    and elapsed + forecast > self.configuration.time_budget_seconds
                ):
                    data['execution_status'] = 'time_budget_exceeded'
                    data['skipped_evaluations'] = [
                        {**task, 'reason': 'projected_time_budget_exceeded'}
                        for task in pending
                    ]
                    self._checkpoint(data)
                    break
                if self.stop_requested:
                    data['execution_status'] = 'interrupted'
                    self._checkpoint(data)
                    break

                task = pending.pop(0)
                started = datetime.now(timezone.utc)
                started_perf = perf_counter()
                result = self.evaluate_task(task)
                elapsed_override = result.pop('_elapsed_seconds', None)
                elapsed_seconds = (
                    float(elapsed_override)
                    if elapsed_override is not None
                    else perf_counter() - started_perf
                )
                objective = float(result['objective'])
                gradient = np.asarray(result['gradient'], dtype=float)
                original_gradient = np.asarray(
                    data['original_result']['gradient'], dtype=float
                )
                if gradient.shape != original_gradient.shape:
                    raise BiogemeError(
                        'Diagnostic gradient shape does not match the original '
                        f'gradient: {gradient.shape} != {original_gradient.shape}.'
                    )
                gradient_difference = gradient - original_gradient
                objective_difference = abs(
                    objective - float(data['original_result']['objective'])
                )
                relative_difference = objective_difference / max(
                    abs(float(data['original_result']['objective'])),
                    np.finfo(float).eps,
                )
                completed_item = {
                    **task,
                    'objective': objective,
                    'gradient': [float(value) for value in gradient],
                    'objective_difference': objective_difference,
                    'relative_objective_difference': relative_difference,
                    'gradient_difference': [
                        float(value) for value in gradient_difference
                    ],
                    'gradient_linf_difference': float(
                        np.linalg.norm(gradient_difference, ord=np.inf)
                    ),
                    'gradient_l2_difference': float(
                        np.linalg.norm(gradient_difference, ord=2)
                    ),
                    'elapsed_seconds': elapsed_seconds,
                    'started_at': started.isoformat(timespec='seconds'),
                    'completed_at': utc_now(),
                    **{
                        key: value
                        for key, value in result.items()
                        if key not in {'objective', 'gradient'}
                    },
                }
                data['completed_evaluations'].append(completed_item)
                data['limitations'] = sorted(
                    set(data['limitations'])
                    | set(completed_item.get('limitations', []))
                )
                data['timing']['elapsed_seconds'] = sum(
                    float(item['elapsed_seconds'])
                    for item in data['completed_evaluations']
                )
                data['timing']['projected_remaining_seconds'] = (
                    forecast_remaining_seconds(
                        completed=list(data['completed_evaluations']),
                        pending=pending,
                        safety_factor=self.configuration.safety_factor,
                    )
                )
                self._checkpoint(data)
                if self.stop_requested:
                    data['execution_status'] = 'interrupted'
                    self._checkpoint(data)
                    break
            else:
                data['execution_status'] = 'completed'
                data['timing']['projected_remaining_seconds'] = 0.0
                self._checkpoint(data)
        except KeyboardInterrupt:
            data['execution_status'] = 'interrupted'
            self._checkpoint(data)
        except Exception as error:
            data['execution_status'] = 'failed'
            data['failure'] = f'{type(error).__name__}: {error}'
            self._checkpoint(data)
            raise
        finally:
            if handler_installed:
                signal.signal(signal.SIGINT, old_handler)

        return MonteCarloDiagnosticResult(
            data=data,
            yaml_file=self.yaml_file,
            markdown_file=self.markdown_file,
        )


def make_diagnostic_evaluator(
    model_elements: ModelElements,
    draw_types: dict[str, str],
    variable_names: list[str],
    estimated_parameters: dict[str, float],
    numerically_safe: bool,
    user_generators: dict[str, Any] | None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a task evaluator that caches JAX evaluators by draw shape."""
    evaluator_cache: dict[int, CompiledFormulaEvaluator] = {}

    def evaluate(task: dict[str, Any]) -> dict[str, Any]:
        draw_count = int(task['draw_count'])
        draws, methods, limitations = generate_diagnostic_draws(
            draw_types=draw_types,
            variable_names=variable_names,
            sample_size=model_elements.sample_size,
            number_of_draws=draw_count,
            seed=int(task['seed']),
            user_generators=user_generators,
        )
        evaluator = evaluator_cache.get(draw_count)
        if evaluator is None:
            diagnostic_draws = DrawsManagement(
                sample_size=model_elements.sample_size,
                number_of_draws=draw_count,
                user_generators=user_generators,
            )
            diagnostic_draws.draws = draws
            diagnostic_draws.draw_types = dict(draw_types)
            diagnostic_elements = copy.copy(model_elements)
            diagnostic_elements.number_of_draws = draw_count
            diagnostic_elements._draws_management = diagnostic_draws
            evaluator = CompiledFormulaEvaluator(
                model_elements=diagnostic_elements,
                second_derivatives_mode=SecondDerivativesMode.NEVER,
                numerically_safe=numerically_safe,
            )
            evaluator_cache[draw_count] = evaluator
        else:
            evaluator.draws_jax = jnp.asarray(draws, dtype=JAX_FLOAT)
        output = evaluator.evaluate(
            the_betas=estimated_parameters,
            gradient=True,
            hessian=False,
            bhhh=False,
        )
        if output.gradient is None:
            raise BiogemeError('The diagnostic evaluator did not return a gradient.')
        return {
            'objective': float(output.function),
            'gradient': [float(value) for value in output.gradient],
            'draw_design_methods': methods,
            'limitations': limitations,
        }

    return evaluate


def run_monte_carlo_diagnostic(
    estimation_results: EstimationResults,
    model_elements: ModelElements,
    configuration: MonteCarloDiagnosticConfiguration,
    yaml_file: Path,
    markdown_file: Path,
    numerically_safe: bool,
    user_generators: dict[str, Any] | None,
    configured_seed: int,
    resume: bool = True,
) -> MonteCarloDiagnosticResult:
    """Run the diagnostic using a fixed existing estimation result."""
    raw = estimation_results.raw_estimation_results
    if not raw.optimization_complete:
        raise BiogemeError(
            'The estimation result is an incomplete optimization checkpoint. '
            'Complete or resume the estimation before running the Monte Carlo '
            'draw-stability diagnostic.'
        )
    estimated_parameters = estimation_results.get_beta_values()
    requires_draws = model_elements.expressions_registry.requires_draws
    if bool(raw.monte_carlo) != bool(requires_draws):
        raise BiogemeError(
            'The reconstructed model and the estimation result disagree about '
            'whether Monte Carlo integration is used.'
        )
    draw_types = model_elements.expressions_registry.draw_types()
    if dict(raw.types_of_draws or {}) != draw_types:
        raise BiogemeError(
            'Draw types in the reconstructed model do not match the estimation '
            'result. Reconstruct the exact estimated specification.'
        )
    gradient = raw.gradient
    if requires_draws and (gradient is None or len(gradient) != len(raw.beta_names)):
        raise BiogemeError(
            'The estimation result does not contain a complete final gradient. '
            'Complete post-estimation gradient calculation before running the '
            'Monte Carlo draw-stability diagnostic.'
        )
    model_parameter_names = list(model_elements.expressions_registry.free_betas_names)
    gradient_by_name = (
        {} if gradient is None else dict(zip(raw.beta_names, gradient, strict=True))
    )
    baseline_gradient = (
        []
        if gradient is None
        else [float(gradient_by_name[name]) for name in model_parameter_names]
    )

    output_baseline = {
        'model_name': raw.model_name,
        'original_number_of_draws': int(raw.number_of_draws),
        'draw_types': dict(draw_types),
        'estimated_parameters': {
            name: float(estimated_parameters[name]) for name in model_parameter_names
        },
        'original_result': {
            'objective': float(raw.final_log_likelihood),
            'gradient': baseline_gradient,
        },
        'model_metadata': {
            'data_name': raw.data_name,
            'sample_size': int(raw.sample_size),
            'number_of_observations': int(raw.number_of_observations),
            'parameter_names': model_parameter_names,
        },
    }

    if not requires_draws:
        data = {
            'schema_version': SCHEMA_VERSION,
            **output_baseline,
            'created_at': utc_now(),
            'updated_at': utc_now(),
            'execution_status': 'completed',
            'diagnostic_conclusion': 'not_applicable',
            'recommendation': 'not_applicable',
            'configuration': configuration.as_dict(),
            'base_seed': None,
            'planned_evaluations': [],
            'completed_evaluations': [],
            'skipped_evaluations': [],
            'timing': {
                'elapsed_seconds': 0.0,
                'projected_remaining_seconds': 0.0,
            },
            'motivation_codes': ['model_has_no_monte_carlo_integration'],
            'limitations': [],
        }
        atomic_write_yaml(yaml_file, data)
        atomic_write_text(markdown_file, generate_markdown_report(data))
        return MonteCarloDiagnosticResult(data, yaml_file, markdown_file)

    schedule = build_draw_schedule(
        original_draws=int(raw.number_of_draws),
        configuration=configuration,
        draw_types=draw_types,
    )
    existing_base_seed: int | None = None
    if resume and yaml_file.exists():
        with yaml_file.open('r', encoding='utf-8') as stream:
            existing = yaml.safe_load(stream)
        if isinstance(existing, dict) and existing.get('base_seed') is not None:
            existing_base_seed = int(existing['base_seed'])
    base_seed = (
        existing_base_seed
        if existing_base_seed is not None
        else (
            (configured_seed + 1_000_003) % (2**32)
            if configured_seed > 0
            else secrets.randbelow(2**32)
        )
    )
    planned_tasks = build_tasks(
        schedule=schedule,
        replications=configuration.replications,
        original_draws=int(raw.number_of_draws),
        draw_types=draw_types,
        base_seed=base_seed,
    )
    evaluate_task = make_diagnostic_evaluator(
        model_elements=model_elements,
        draw_types=draw_types,
        variable_names=model_elements.expressions_registry.draws_names,
        estimated_parameters=estimated_parameters,
        numerically_safe=numerically_safe,
        user_generators=user_generators,
    )
    runner = MonteCarloDiagnosticRunner(
        baseline=output_baseline,
        configuration=configuration,
        planned_tasks=planned_tasks,
        evaluate_task=evaluate_task,
        yaml_file=yaml_file,
        markdown_file=markdown_file,
        base_seed=base_seed,
        resume=resume,
    )
    return runner.run()
