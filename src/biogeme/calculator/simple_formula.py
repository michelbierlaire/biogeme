from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .single_formula import evaluate_expression_per_row, evaluate_formula
from biogeme.model_elements import ModelElements

if TYPE_CHECKING:
    from biogeme.database import Database
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
    )
