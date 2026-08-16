#!/usr/bin/env python3
"""Shared support for the Biogeme release-comparison estimators.

The model definitions in this module are copied from the Biogeme 3.3.4
Swissmetro examples.  The generated entry-point files select the small API
adapter needed by each target release.  Keeping the definitions here prevents
the nine estimators from drifting apart while still leaving nine explicit
scripts that can be submitted independently on JED.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / 'docs' / 'source' / 'examples' / 'swissmetro' / 'swissmetro.dat'

MODEL_DRAW_COUNTS = {
    # The 3.3.3 full analytical-Hessian implementation needs substantially
    # less memory than the original 10,000-draw benchmark allowed.  Keep this
    # value common to all releases so that the comparison remains fair.
    'b05a_normal_mixture': 2_000,
    'b11a_cnl': None,
    'b12_panel': 5_000,
}


@dataclass(frozen=True)
class BiogemeApi:
    """The names needed by the common model definitions."""

    BIOGEME: Any
    Beta: Any
    Draws: Any
    MonteCarlo: Any
    PanelLikelihoodTrajectory: Any
    log: Any
    logit: Any
    logcnl: Any
    OneNestForCrossNestedLogit: Any
    NestsForCrossNestedLogit: Any
    legacy: bool


def load_api(*, legacy: bool) -> BiogemeApi:
    """Load the common API from either the legacy or modern distribution."""
    if legacy:
        import biogeme.biogeme as bio
        from biogeme import models
        from biogeme.expressions import (
            Beta,
            MonteCarlo,
            PanelLikelihoodTrajectory,
            bioDraws,
            log,
        )
        from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit

        return BiogemeApi(
            BIOGEME=bio.BIOGEME,
            Beta=Beta,
            Draws=bioDraws,
            MonteCarlo=MonteCarlo,
            PanelLikelihoodTrajectory=PanelLikelihoodTrajectory,
            log=log,
            logit=models.logit,
            logcnl=models.logcnl,
            OneNestForCrossNestedLogit=OneNestForCrossNestedLogit,
            NestsForCrossNestedLogit=NestsForCrossNestedLogit,
            legacy=True,
        )

    from biogeme.biogeme import BIOGEME
    from biogeme.expressions import (
        Beta,
        Draws,
        MonteCarlo,
        PanelLikelihoodTrajectory,
        log,
    )
    from biogeme.models import logcnl, logit
    from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit

    return BiogemeApi(
        BIOGEME=BIOGEME,
        Beta=Beta,
        Draws=Draws,
        MonteCarlo=MonteCarlo,
        PanelLikelihoodTrajectory=PanelLikelihoodTrajectory,
        log=log,
        logit=logit,
        logcnl=logcnl,
        OneNestForCrossNestedLogit=OneNestForCrossNestedLogit,
        NestsForCrossNestedLogit=NestsForCrossNestedLogit,
        legacy=False,
    )


def _database_class() -> Any:
    """Return Database across the two import layouts."""
    try:
        from biogeme.database import Database

        return Database
    except ImportError:  # Biogeme 3.2.x layout.
        import biogeme.database as database_module

        return database_module.Database


def _variables(names: list[str]) -> SimpleNamespace:
    from biogeme.expressions import Variable

    return SimpleNamespace(**{name: Variable(name) for name in names})


def load_data(*, model: str, data_path: Path) -> SimpleNamespace:
    """Load and prepare the Swissmetro data used by the 3.3.4 examples."""
    import pandas as pd

    Database = _database_class()
    dataframe = pd.read_csv(data_path, sep='\t')
    database = Database('swissmetro', dataframe)

    if model == 'b12_panel':
        variable_names = [
            'PURPOSE',
            'CHOICE',
            'GA',
            'TRAIN_CO',
            'CAR_AV',
            'SP',
            'TRAIN_AV',
            'TRAIN_TT',
            'SM_TT',
            'CAR_TT',
            'CAR_CO',
            'SM_CO',
            'SM_AV',
            'ID',
        ]
    else:
        variable_names = [
            'PURPOSE',
            'CHOICE',
            'GA',
            'TRAIN_CO',
            'CAR_AV',
            'SP',
            'TRAIN_AV',
            'TRAIN_TT',
            'TRAIN_HE',
            'SM_TT',
            'SM_CO',
            'SM_HE',
            'CAR_TT',
            'CAR_CO',
            'SM_AV',
        ]

    variables = _variables(variable_names)
    exclude = ((variables.PURPOSE != 1) * (variables.PURPOSE != 3) + (variables.CHOICE == 0)) > 0
    database.remove(exclude)

    sm_cost = database.define_variable('SM_COST', variables.SM_CO * (variables.GA == 0))
    train_cost = database.define_variable(
        'TRAIN_COST', variables.TRAIN_CO * (variables.GA == 0)
    )
    car_av_sp = database.define_variable('CAR_AV_SP', variables.CAR_AV * (variables.SP != 0))
    train_av_sp = database.define_variable(
        'TRAIN_AV_SP', variables.TRAIN_AV * (variables.SP != 0)
    )
    train_tt_scaled = database.define_variable('TRAIN_TT_SCALED', variables.TRAIN_TT / 100)
    train_cost_scaled = database.define_variable('TRAIN_COST_SCALED', train_cost / 100)
    sm_tt_scaled = database.define_variable('SM_TT_SCALED', variables.SM_TT / 100)
    sm_cost_scaled = database.define_variable('SM_COST_SCALED', sm_cost / 100)
    car_tt_scaled = database.define_variable('CAR_TT_SCALED', variables.CAR_TT / 100)
    car_co_scaled = database.define_variable('CAR_CO_SCALED', variables.CAR_CO / 100)

    result = vars(variables)
    result.update(
        database=database,
        CAR_AV_SP=car_av_sp,
        TRAIN_AV_SP=train_av_sp,
        TRAIN_TT_SCALED=train_tt_scaled,
        TRAIN_COST_SCALED=train_cost_scaled,
        SM_TT_SCALED=sm_tt_scaled,
        SM_COST_SCALED=sm_cost_scaled,
        CAR_TT_SCALED=car_tt_scaled,
        CAR_CO_SCALED=car_co_scaled,
    )
    if model == 'b12_panel':
        database.panel('ID')
    return SimpleNamespace(**result)


def build_b05(api: BiogemeApi, data: SimpleNamespace) -> Any:
    """Build the 3.3.4 ``plot_b05a_normal_mixture.py`` log likelihood."""
    asc_car = api.Beta('asc_car', 0, None, None, 0)
    asc_train = api.Beta('asc_train', 0, None, None, 0)
    asc_sm = api.Beta('asc_sm', 0, None, None, 1)
    b_cost = api.Beta('b_cost', 0, None, None, 0)
    b_time = api.Beta('b_time', 0, None, None, 0)
    b_time_s = api.Beta('b_time_s', 1, None, None, 0)
    b_time_rnd = b_time + b_time_s * api.Draws('b_time_rnd', 'NORMAL')

    v = {
        1: asc_train + b_time_rnd * data.TRAIN_TT_SCALED + b_cost * data.TRAIN_COST_SCALED,
        2: asc_sm + b_time_rnd * data.SM_TT_SCALED + b_cost * data.SM_COST_SCALED,
        3: asc_car + b_time_rnd * data.CAR_TT_SCALED + b_cost * data.CAR_CO_SCALED,
    }
    availability = {1: data.TRAIN_AV_SP, 2: data.SM_AV, 3: data.CAR_AV_SP}
    return api.log(api.MonteCarlo(api.logit(v, availability, data.CHOICE)))


def build_b11(api: BiogemeApi, data: SimpleNamespace) -> Any:
    """Build the 3.3.4 ``plot_b11a_cnl.py`` log likelihood."""
    asc_car = api.Beta('asc_car', 0, None, None, 0)
    asc_train = api.Beta('asc_train', 0, None, None, 0)
    asc_sm = api.Beta('asc_sm', 0, None, None, 1)
    b_time_swissmetro = api.Beta('b_time_swissmetro', 0, None, None, 0)
    b_time_train = api.Beta('b_time_train', 0, None, None, 0)
    b_time_car = api.Beta('b_time_car', 0, None, None, 0)
    b_cost = api.Beta('b_cost', 0, None, None, 0)
    b_headway_swissmetro = api.Beta('b_headway_swissmetro', 0, None, None, 0)
    b_headway_train = api.Beta('b_headway_train', 0, None, None, 0)
    ga_train = api.Beta('ga_train', 0, None, None, 0)
    ga_swissmetro = api.Beta('ga_swissmetro', 0, None, None, 0)
    existing_nest_parameter = api.Beta('existing_nest_parameter', 1, 1, 5, 0)
    public_nest_parameter = api.Beta('public_nest_parameter', 1, 1, 5, 0)
    alpha_existing = api.Beta('alpha_existing', 0.5, 0, 1, 0)
    alpha_public = 1 - alpha_existing

    v = {
        1: (
            asc_train
            + b_time_train * data.TRAIN_TT_SCALED
            + b_cost * data.TRAIN_COST_SCALED
            + b_headway_train * data.TRAIN_HE
            + ga_train * data.GA
        ),
        2: (
            asc_sm
            + b_time_swissmetro * data.SM_TT_SCALED
            + b_cost * data.SM_COST_SCALED
            + b_headway_swissmetro * data.SM_HE
            + ga_swissmetro * data.GA
        ),
        3: asc_car + b_time_car * data.CAR_TT_SCALED + b_cost * data.CAR_CO_SCALED,
    }
    availability = {1: data.TRAIN_AV_SP, 2: data.SM_AV, 3: data.CAR_AV_SP}
    nest_existing = api.OneNestForCrossNestedLogit(
        nest_param=existing_nest_parameter,
        dict_of_alpha={1: alpha_existing, 2: 0.0, 3: 1.0},
        name='existing',
    )
    nest_public = api.OneNestForCrossNestedLogit(
        nest_param=public_nest_parameter,
        dict_of_alpha={1: alpha_public, 2: 1.0, 3: 0.0},
        name='public',
    )
    nests = api.NestsForCrossNestedLogit(
        choice_set=[1, 2, 3], tuple_of_nests=(nest_existing, nest_public)
    )
    return api.logcnl(v, availability, nests, data.CHOICE)


def build_b12(api: BiogemeApi, data: SimpleNamespace) -> Any:
    """Build the 3.3.4 ``plot_b12_panel.py`` log likelihood."""
    b_cost = api.Beta('b_cost', 0, None, 0, 0)
    b_time = api.Beta('b_time', 0, None, 0, 0)
    b_time_s = api.Beta('b_time_s', 1, 1.0e-5, None, 0)
    b_time_rnd = b_time + b_time_s * api.Draws('b_time_rnd', 'NORMAL_ANTI')

    asc_car = api.Beta('asc_car', 0, None, None, 0)
    asc_car_s = api.Beta('asc_car_s', 1, 1.0e-5, None, 0)
    asc_car_rnd = asc_car + asc_car_s * api.Draws('asc_car_rnd', 'NORMAL_ANTI')
    asc_train = api.Beta('asc_train', 0, None, None, 0)
    asc_train_s = api.Beta('asc_train_s', 1, 1.0e-5, None, 0)
    asc_train_rnd = asc_train + asc_train_s * api.Draws('asc_train_rnd', 'NORMAL_ANTI')
    asc_sm = api.Beta('asc_sm', 0, None, None, 0)
    asc_sm_s = api.Beta('asc_sm_s', 1, 1.0e-5, None, 0)
    asc_sm_rnd = asc_sm + asc_sm_s * api.Draws('asc_sm_rnd', 'NORMAL_ANTI')

    v = {
        1: asc_train_rnd + b_time_rnd * data.TRAIN_TT_SCALED + b_cost * data.TRAIN_COST_SCALED,
        2: asc_sm_rnd + b_time_rnd * data.SM_TT_SCALED + b_cost * data.SM_COST_SCALED,
        3: asc_car_rnd + b_time_rnd * data.CAR_TT_SCALED + b_cost * data.CAR_CO_SCALED,
    }
    availability = {1: data.TRAIN_AV_SP, 2: data.SM_AV, 3: data.CAR_AV_SP}
    one_observation = api.logit(v, availability, data.CHOICE)
    trajectory = api.PanelLikelihoodTrajectory(one_observation)
    return api.log(api.MonteCarlo(trajectory))


MODEL_BUILDERS: dict[str, Callable[[BiogemeApi, SimpleNamespace], Any]] = {
    'b05a_normal_mixture': build_b05,
    'b11a_cnl': build_b11,
    'b12_panel': build_b12,
}


def _modern_kwargs(model: str, *, release: str) -> dict[str, Any]:
    """Parameters used by Biogeme 3.3.x, independent of cwd TOML files."""
    kwargs: dict[str, Any] = {
        'generate_html': False,
        'generate_yaml': False,
        'generate_netcdf': False,
        'save_iterations': False,
        'optimization_algorithm': 'automatic',
        'max_iterations': 1000,
        'number_of_threads': 1,
        'tolerance': 6.055454452393343e-06,
    }
    if model == 'b05a_normal_mixture':
        kwargs.update(calculating_second_derivatives='analytical')
        if release == '3.3.4':
            kwargs['analytical_hessian_mode'] = 'automatic'
    elif model == 'b11a_cnl':
        kwargs.update(calculating_second_derivatives='analytical')
        if release == '3.3.4':
            kwargs['analytical_hessian_mode'] = 'full'
    elif model == 'b12_panel':
        kwargs.update(calculating_second_derivatives='never')
    return kwargs


def _legacy_kwargs(model: str) -> dict[str, Any]:
    """Parameters accepted by Biogeme 3.2.14 for the same policies."""
    kwargs: dict[str, Any] = {
        'generate_html': False,
        'generate_pickle': False,
        'save_iterations': False,
        'optimization_algorithm': 'automatic',
        'max_iterations': 1000,
        'number_of_threads': 1,
        'tolerance': 6.055454452393343e-06,
    }
    # The legacy optimizer decides whether to use Hessians from model
    # complexity.  The value is relevant only for explicit simple_bounds;
    # retaining it here documents the intended mapping for the panel case.
    if model == 'b12_panel':
        kwargs['second_derivatives'] = 0.0
    return kwargs


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, (int, float)):
        return value
    if hasattr(value, 'item'):
        try:
            item = value.item()
            if isinstance(item, (int, float)):
                return item
        except (TypeError, ValueError):
            pass
    return None


def _message(messages: Any, *keys: str) -> Any:
    if messages is None:
        return None
    for key in keys:
        if hasattr(messages, 'get'):
            value = messages.get(key)
            if value is not None:
                return value
        value = getattr(messages, key, None)
        if value is not None:
            return value
    return None


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, 'total_seconds'):
        return float(value.total_seconds())
    number = _number(value)
    return float(number) if number is not None else None


def extract_result(result: Any) -> dict[str, Any]:
    """Extract comparable diagnostics from old and modern result objects."""
    raw = getattr(result, 'raw_estimation_results', None)
    if raw is None:
        raw = getattr(result, 'data', None)

    beta_values = result.get_beta_values()
    final_log_likelihood = getattr(result, 'final_log_likelihood', None)
    if final_log_likelihood is None and raw is not None:
        final_log_likelihood = getattr(raw, 'logLike', None)

    converged = getattr(result, 'algorithm_has_converged', None)
    if callable(converged):
        converged = converged()
    if converged is None and raw is not None:
        converged = getattr(raw, 'convergence', None)

    messages = None
    if raw is not None:
        messages = getattr(raw, 'optimization_messages', None)
        if messages is None:
            messages = getattr(raw, 'optimizationMessages', None)

    return {
        'final_log_likelihood': _number(final_log_likelihood),
        'estimated_parameters': {
            str(name): _number(value) for name, value in beta_values.items()
        },
        'converged': bool(converged),
        'optimization_time_seconds': _seconds(
            _message(messages, 'Optimization time', 'optimization_time')
        ),
        'iterations': _number(
            _message(messages, 'Number of iterations', 'number_of_iterations')
        ),
        'function_evaluations': _number(
            _message(messages, 'Number of function evaluations', 'function_evaluations')
        ),
        'gradient_evaluations': _number(
            _message(messages, 'Number of gradient evaluations', 'gradient_evaluations')
        ),
        'hessian_evaluations': _number(
            _message(messages, 'Number of hessian evaluations', 'hessian_evaluations')
        ),
    }


def _timed_warm_evaluation(function: Callable[[], Any]) -> float:
    """Return the time of one warm evaluation of ``function``.

    A first JAX call can include compilation.  The benchmark's total wall
    time deliberately includes that cost, while these supplementary numbers
    describe the steady-state cost of one objective/derivative evaluation.
    The untimed call also ensures that asynchronous JAX work is completed
    before the measured call starts.
    """
    function()
    started = perf_counter()
    function()
    return perf_counter() - started


def measure_evaluation_times(
    biogeme: Any, result: Any, *, legacy: bool
) -> dict[str, float | None | str]:
    """Measure one warm likelihood/gradient and Hessian evaluation.

    The legacy C++ API and the modern JAX evaluator expose different entry
    points, so this adapter keeps the benchmark scripts independent of those
    details.  A Hessian time is reported only when the modern configuration
    requested second derivatives; otherwise it is explicitly marked as not
    requested rather than computed as an extra, unrepresentative operation.
    """
    beta_values = result.get_beta_values()

    if legacy:
        names = list(biogeme.free_beta_names)
        beta_vector = np.asarray([beta_values[name] for name in names], dtype=float)

        def likelihood_gradient() -> Any:
            return biogeme.calculate_likelihood_and_derivatives(
                beta_vector, scaled=False, hessian=False, bhhh=False
            )

        def likelihood_gradient_hessian() -> Any:
            return biogeme.calculate_likelihood_and_derivatives(
                beta_vector, scaled=False, hessian=True, bhhh=False
            )

        return {
            'likelihood_gradient_seconds': _timed_warm_evaluation(
                likelihood_gradient
            ),
            'likelihood_gradient_hessian_seconds': _timed_warm_evaluation(
                likelihood_gradient_hessian
            ),
            'evaluation_timing_method': 'warm evaluation after one untimed call',
        }

    def likelihood_gradient() -> Any:
        return biogeme.function_evaluator.evaluate(
            the_betas=beta_values,
            gradient=True,
            hessian=False,
            bhhh=False,
        )

    gradient_seconds = _timed_warm_evaluation(likelihood_gradient)
    second_derivatives_mode = getattr(biogeme, 'second_derivatives_mode', None)
    mode_value = getattr(second_derivatives_mode, 'value', second_derivatives_mode)
    if mode_value == 'never':
        hessian_seconds = None
        timing_method = 'warm evaluation after one untimed call; Hessian not requested'
    else:

        def likelihood_gradient_hessian() -> Any:
            return biogeme.function_evaluator.evaluate(
                the_betas=beta_values,
                gradient=True,
                hessian=True,
                bhhh=False,
            )

        hessian_seconds = _timed_warm_evaluation(likelihood_gradient_hessian)
        timing_method = 'warm evaluation after one untimed call'

    return {
        'likelihood_gradient_seconds': gradient_seconds,
        'likelihood_gradient_hessian_seconds': hessian_seconds,
        'evaluation_timing_method': timing_method,
    }


def parse_arguments(*, release: str, model: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f'Run {model} with Biogeme {release}.')
    parser.add_argument(
        '--output',
        type=Path,
        help='JSON output path; if omitted, only the JSON record on stdout is produced.',
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path(os.environ.get('BIOGEME_BENCHMARK_DATA', DEFAULT_DATA_PATH)),
        help='Swissmetro data file (default: the repository data file).',
    )
    return parser.parse_args()


def run_case(*, release: str, model: str, legacy: bool) -> int:
    if model not in MODEL_BUILDERS:
        raise ValueError(f'Unknown benchmark model: {model}')

    args = parse_arguments(release=release, model=model)
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f'Swissmetro data file not found: {data_path}')

    api = load_api(legacy=legacy)
    data = load_data(model=model, data_path=data_path)
    log_probability = MODEL_BUILDERS[model](api, data)
    number_of_draws = MODEL_DRAW_COUNTS[model]

    kwargs = (
        _legacy_kwargs(model)
        if legacy
        else _modern_kwargs(model, release=release)
    )
    if number_of_draws is not None:
        kwargs['number_of_draws'] = number_of_draws
    kwargs['seed'] = 1223
    # Do not let a release read or generate the optional working-directory
    # ``biogeme.toml``. It is ignored by Git and is absent on a clean JED
    # checkout; older tomlkit versions can also fail while generating their
    # default file. An in-memory Parameters object gives all releases the
    # same explicit configuration without touching the filesystem.
    from biogeme.parameters import Parameters

    parameters = Parameters()
    biogeme = api.BIOGEME(
        data.database,
        log_probability,
        parameters=parameters,
        **kwargs,
    )
    model_name = f'release_benchmark_{release.replace(".", "_")}_{model}'
    if legacy:
        biogeme.modelName = model_name
    else:
        biogeme.model_name = model_name

    started_at = datetime.now(timezone.utc)
    start = perf_counter()
    if legacy:
        result = biogeme.estimate(recycle=False)
    else:
        result = biogeme.estimate()
    elapsed = perf_counter() - start
    evaluation_times = measure_evaluation_times(biogeme, result, legacy=legacy)

    record = {
        'schema_version': 2,
        'release': release,
        'model': model,
        'model_name': model_name,
        'legacy_api': legacy,
        'python': sys.version,
        'python_executable': sys.executable,
        'biogeme_module': str(__import__('biogeme').__file__),
        'biogeme_distribution_version': importlib.metadata.version('biogeme'),
        'data_path': str(data_path),
        'number_of_draws': number_of_draws,
        'seed': 1223,
        'wall_time_seconds': elapsed,
        **evaluation_times,
        'started_at_utc': started_at.isoformat(timespec='seconds'),
        'configuration': {
            key: value
            for key, value in kwargs.items()
            if key not in {'seed', 'number_of_draws'}
        },
    }
    record.update(extract_result(result))

    serialized = json.dumps(record, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    print(f'BENCHMARK_RESULT {json.dumps(record, sort_keys=True)}')
    return 0
