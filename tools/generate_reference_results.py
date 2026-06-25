"""Generate golden reference results for single_formula.py.

Run this script once with the trusted/reference implementation, then commit the
generated `.npz` files under tests/data/reference/.

It covers:
- function only
- function + gradient
- function + gradient + Hessian
- function + gradient + BHHH
- function + gradient + Hessian + BHHH
- weighted and unweighted likelihoods
- JIT and no-JIT
- analytical/autodiff and finite-difference Hessian modes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Beta, Variable, exp, log
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import FlatPanelAdapter, ModelElements, RegularAdapter
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.version import get_version

REFERENCE_DIR = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'reference'
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    expression: object
    data: pd.DataFrame
    betas: dict[str, float]
    weight: object | None = None


def make_adapter(database: Database):
    return (
        FlatPanelAdapter(database=database)
        if database.is_panel()
        else RegularAdapter(database=database)
    )


def build_cases() -> list[ReferenceCase]:
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

    return [
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


def evaluate_case(
    case: ReferenceCase,
    *,
    use_jit: bool,
    second_derivatives_mode: SecondDerivativesMode,
    gradient: bool,
    hessian: bool,
    bhhh: bool,
):
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

    return {
        'function': np.asarray(output.function, dtype=float),
        'gradient': (
            np.asarray(output.gradient, dtype=float)
            if output.gradient is not None
            else np.asarray([], dtype=float)
        ),
        'hessian': (
            np.asarray(output.hessian, dtype=float)
            if output.hessian is not None
            else np.asarray([[]], dtype=float)
        ),
        'bhhh': (
            np.asarray(output.bhhh, dtype=float)
            if output.bhhh is not None
            else np.asarray([[]], dtype=float)
        ),
        'beta_names': np.asarray(evaluator.free_betas_names, dtype=str),
        'beta_values': np.asarray(beta_values, dtype=float),
    }


def output_filename(
    case_name: str,
    *,
    use_jit: bool,
    second_derivatives_mode: SecondDerivativesMode,
    gradient: bool,
    hessian: bool,
    bhhh: bool,
) -> Path:
    parts = [
        case_name,
        f'jit_{int(use_jit)}',
        f'mode_{second_derivatives_mode.name.lower()}',
        f'g_{int(gradient)}',
        f'h_{int(hessian)}',
        f'b_{int(bhhh)}',
    ]
    return REFERENCE_DIR / ('__'.join(parts) + '.npz')


def get_biogeme_version() -> str:
    """Return the Biogeme version."""
    return get_version()


def main() -> None:
    biogeme_version = get_biogeme_version()
    cases = build_cases()

    configurations = [
        # function only
        dict(gradient=False, hessian=False, bhhh=False),
        # first derivatives
        dict(gradient=True, hessian=False, bhhh=False),
        # Hessian only path through gradient=True
        dict(gradient=True, hessian=True, bhhh=False),
        # BHHH only
        dict(gradient=True, hessian=False, bhhh=True),
        # Hessian + BHHH
        dict(gradient=True, hessian=True, bhhh=True),
    ]

    derivative_modes = [
        SecondDerivativesMode.ANALYTICAL,
        SecondDerivativesMode.FINITE_DIFFERENCES,
    ]

    for case in cases:
        for use_jit in (False, True):
            for mode in derivative_modes:
                for config in configurations:
                    if (
                        not config['hessian']
                        and mode == SecondDerivativesMode.FINITE_DIFFERENCES
                    ):
                        continue

                    result = evaluate_case(
                        case,
                        use_jit=use_jit,
                        second_derivatives_mode=mode,
                        **config,
                    )

                    filename = output_filename(
                        case.name,
                        use_jit=use_jit,
                        second_derivatives_mode=mode,
                        **config,
                    )

                    np.savez_compressed(
                        filename,
                        case_name=np.asarray(case.name),
                        biogeme_version=np.asarray(biogeme_version),
                        use_jit=np.asarray(use_jit),
                        second_derivatives_mode=np.asarray(mode.name),
                        gradient_requested=np.asarray(config['gradient']),
                        hessian_requested=np.asarray(config['hessian']),
                        bhhh_requested=np.asarray(config['bhhh']),
                        **result,
                    )
                    print(f'written: {filename}')
    print()
    print(f'Reference files generated using Biogeme {biogeme_version}.')


if __name__ == '__main__':
    main()
