"""Regression tests against golden reference files for single_formula.py.

The golden files are generated once with a trusted Biogeme release, typically
an official PyPI version. These tests are then run with the development version
and compare function values, gradients, Hessians, and BHHH matrices.

Important: the tested Biogeme version must be different from the version stored
in the golden files. Otherwise, the test would only compare the implementation
with itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from biogeme.database import Database
from biogeme.expressions import Beta, Variable, exp, log
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import FlatPanelAdapter, ModelElements, RegularAdapter
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.version import get_version

REFERENCE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'reference'


@dataclass(frozen=True)
class ReferenceCase:
    """Information needed to rebuild one golden-reference case."""

    name: str
    expression: Any
    data: pd.DataFrame
    betas: dict[str, float]
    weight: Any | None = None


def make_adapter(database: Database):
    """Create the adapter expected by ModelElements."""
    return (
        FlatPanelAdapter(database=database)
        if database.is_panel()
        else RegularAdapter(database=database)
    )


def build_cases() -> dict[str, ReferenceCase]:
    """Build exactly the same deterministic cases as the generator script."""
    x = Variable('x')
    z = Variable('z')
    w = Variable('weight')

    beta_x = Beta('beta_x', 2.0, None, None, 0)
    beta_z = Beta('beta_z', -1.0, None, None, 0)

    data = pd.DataFrame(
        {
            'x': [1.0, 2.0, 3.0, 4.0],
            'z': [0.5, -1.0, 2.0, 0.0],
            'weight': [1.0, 0.5, 2.0, 1.5],
        }
    )

    cases = [
        ReferenceCase(
            name='linear_unweighted',
            expression=beta_x * x + beta_z * z,
            data=data,
            betas={'beta_x': 2.0, 'beta_z': -1.0},
        ),
        ReferenceCase(
            name='linear_weighted',
            expression=beta_x * x + beta_z * z,
            data=data,
            betas={'beta_x': 2.0, 'beta_z': -1.0},
            weight=w,
        ),
        ReferenceCase(
            name='nonlinear_unweighted',
            expression=log(exp(beta_x * x) + exp(beta_z * z)),
            data=data,
            betas={'beta_x': 0.3, 'beta_z': -0.7},
        ),
        ReferenceCase(
            name='nonlinear_weighted',
            expression=log(exp(beta_x * x) + exp(beta_z * z)),
            data=data,
            betas={'beta_x': 0.3, 'beta_z': -0.7},
            weight=w,
        ),
    ]

    return {case.name: case for case in cases}


def available_reference_files() -> list[Path]:
    """Return all available golden reference files."""
    if not REFERENCE_DIR.exists():
        return []
    return sorted(REFERENCE_DIR.glob('*.npz'))


def scalar_string(npz_file: np.lib.npyio.NpzFile, key: str) -> str:
    """Read a scalar string stored in an npz file."""
    return str(npz_file[key].item())


def scalar_bool(npz_file: np.lib.npyio.NpzFile, key: str) -> bool:
    """Read a scalar boolean stored in an npz file."""
    return bool(npz_file[key].item())


def evaluate_case(
    case: ReferenceCase,
    *,
    use_jit: bool,
    second_derivatives_mode: SecondDerivativesMode,
    gradient: bool,
    hessian: bool,
    bhhh: bool,
):
    """Evaluate one case with the current implementation."""
    database = Database(case.name, case.data)
    adapter = make_adapter(database)

    model_elements = ModelElements.from_expression_and_weight(
        log_like=case.expression,
        weight=case.weight,
        adapter=adapter,
        use_jit=use_jit,
    )

    evaluator = CompiledFormulaEvaluator(
        model_elements=model_elements,
        second_derivatives_mode=second_derivatives_mode,
        numerically_safe=False,
    )

    output = evaluator.evaluate(
        the_betas=case.betas,
        gradient=gradient,
        hessian=hessian,
        bhhh=bhhh,
    )

    beta_values = model_elements.expressions_registry.get_complete_betas_array(
        betas_dict=case.betas
    )

    return (
        output,
        np.asarray(evaluator.free_betas_names, dtype=str),
        np.asarray(beta_values, dtype=float),
    )


def assert_vector_or_matrix_close(
    actual: np.ndarray | None,
    expected: np.ndarray,
    *,
    name: str,
    requested: bool,
    second_derivatives_mode: str,
) -> None:
    """Compare an optional array produced by the evaluator."""
    if not requested:
        assert actual is None, f'{name} should not have been computed.'
        return

    assert actual is not None, f'{name} was requested but is None.'

    # Finite-difference Hessians are slightly less precise than autodiff ones.
    if name == 'hessian' and second_derivatives_mode == 'FINITE_DIFFERENCES':
        rtol = 1.0e-5
        atol = 1.0e-7
    else:
        rtol = 1.0e-10
        atol = 1.0e-10

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f'{name} differs from golden reference.',
    )


reference_files = available_reference_files()


@pytest.mark.skipif(
    not reference_files,
    reason=f'No golden reference files found in {REFERENCE_DIR}',
)
def test_all_golden_files_were_generated_with_same_biogeme_version():
    """All golden files should come from one reference Biogeme version."""
    versions = set()
    for filename in reference_files:
        with np.load(filename, allow_pickle=False) as reference:
            versions.add(scalar_string(reference, 'biogeme_version'))

    assert len(versions) == 1, f'Golden files use multiple Biogeme versions: {versions}'


@pytest.mark.skipif(
    not reference_files,
    reason=f'No golden reference files found in {REFERENCE_DIR}',
)
def test_current_biogeme_version_differs_from_golden_reference_version():
    """The golden files must not have been generated by the version under test."""
    current_version = get_version()
    for filename in reference_files:
        with np.load(filename, allow_pickle=False) as reference:
            reference_version = scalar_string(reference, 'biogeme_version')

        assert current_version != reference_version, (
            f'{filename.name} was generated with Biogeme {reference_version}, '
            f'which is also the version currently under test. Generate the golden '
            f'files with a distinct trusted release, typically the official PyPI '
            f'version, before running this regression test.'
        )


@pytest.mark.skipif(
    not reference_files,
    reason=f'No golden reference files found in {REFERENCE_DIR}',
)
@pytest.mark.parametrize('filename', reference_files, ids=lambda path: path.stem)
def test_single_formula_matches_golden_reference_file(filename: Path):
    """Compare current single-formula outputs against one golden file."""
    cases = build_cases()

    with np.load(filename, allow_pickle=False) as reference:
        case_name = scalar_string(reference, 'case_name')
        assert case_name in cases, f'Unknown golden-reference case: {case_name}'

        use_jit = scalar_bool(reference, 'use_jit')
        mode_name = scalar_string(reference, 'second_derivatives_mode')
        second_derivatives_mode = SecondDerivativesMode[mode_name]
        gradient_requested = scalar_bool(reference, 'gradient_requested')
        hessian_requested = scalar_bool(reference, 'hessian_requested')
        bhhh_requested = scalar_bool(reference, 'bhhh_requested')

        output, beta_names, beta_values = evaluate_case(
            cases[case_name],
            use_jit=use_jit,
            second_derivatives_mode=second_derivatives_mode,
            gradient=gradient_requested,
            hessian=hessian_requested,
            bhhh=bhhh_requested,
        )

        np.testing.assert_array_equal(beta_names, reference['beta_names'])
        np.testing.assert_allclose(
            beta_values,
            reference['beta_values'],
            rtol=0.0,
            atol=0.0,
            err_msg='Beta values differ from golden reference.',
        )

        assert output.function == pytest.approx(float(reference['function']))

        assert_vector_or_matrix_close(
            output.gradient,
            reference['gradient'],
            name='gradient',
            requested=gradient_requested,
            second_derivatives_mode=mode_name,
        )
        assert_vector_or_matrix_close(
            output.hessian,
            reference['hessian'],
            name='hessian',
            requested=hessian_requested,
            second_derivatives_mode=mode_name,
        )
        assert_vector_or_matrix_close(
            output.bhhh,
            reference['bhhh'],
            name='bhhh',
            requested=bhhh_requested,
            second_derivatives_mode=mode_name,
        )
