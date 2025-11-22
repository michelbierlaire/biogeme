from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from biogeme.database import Database
from biogeme.model_elements import ModelElements
from biogeme.second_derivatives import SecondDerivativesMode
from .function_call import (
    CallableExpression,
    CompiledFormulaEvaluator,
    function_from_compiled_formula,
)
from .single_formula import evaluate_formula, evaluate_model_per_row
from ..model_elements.database_adapter import RegularAdapter

if TYPE_CHECKING:
    from biogeme.expressions import Expression


def evaluate_simple_expression_per_row(
    expression: Expression,
    database: Database,
    numerically_safe: bool,
    use_jit: bool,
    second_derivatives_mode: SecondDerivativesMode,
) -> np.ndarray:
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=use_jit,
    )
    return evaluate_model_per_row(
        model_elements=model_elements,
        the_betas={},
        numerically_safe=numerically_safe,
        second_derivatives_mode=second_derivatives_mode,
    )


def evaluate_simple_expression(
    expression: Expression,
    database: Database | None,
    numerically_safe: bool,
    use_jit: bool,
) -> float:
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=use_jit,
    )

    return evaluate_formula(
        model_elements=model_elements,
        the_betas={},
        second_derivatives_mode=SecondDerivativesMode.NEVER,
        numerically_safe=numerically_safe,
    )


def create_function_simple_expression(
    expression: Expression,
    numerically_safe: bool,
    use_jit: bool,
    named_output: bool = False,
    database: Database | None = None,
) -> CallableExpression:
    if database is None:
        database = Database.dummy_database()
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=use_jit,
    )

    the_evaluator = CompiledFormulaEvaluator(
        model_elements=model_elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=numerically_safe,
    )
    betas = model_elements.expressions_registry.free_betas_init_values
    return function_from_compiled_formula(
        the_compiled_function=the_evaluator, the_betas=betas, named_output=named_output
    )
