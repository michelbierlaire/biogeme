from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from biogeme.database import Database
from biogeme.model_elements import ModelElements
from .function_call import (
    CallableExpression,
    CompiledFormulaEvaluator,
    function_from_expression,
)
from .single_formula import evaluate_expression_per_row, evaluate_formula

if TYPE_CHECKING:
    from biogeme.expressions import Expression


def evaluate_simple_expression_per_row(
    expression: Expression, database: Database
) -> np.ndarray:
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression, weight=None, database=database
    )
    return evaluate_expression_per_row(
        model_elements=model_elements,
        the_betas={},
    )


def evaluate_simple_expression(expression: Expression, database: Database) -> float:
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression, weight=None, database=database
    )

    return evaluate_formula(
        model_elements=model_elements,
        the_betas={},
        avoid_analytical_second_derivatives=False,
    )


def create_function_simple_expression(
    expression: Expression,
    database: Database | None = None,
) -> CallableExpression:
    if database is None:
        database = Database.dummy_database()
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression, weight=None, database=database
    )

    the_evaluator = CompiledFormulaEvaluator(
        model_elements=model_elements, avoid_analytical_second_derivatives=False
    )
    betas = model_elements.expressions_registry.free_betas_init_values
    return function_from_expression(
        the_compiled_function=the_evaluator, the_betas=betas
    )
