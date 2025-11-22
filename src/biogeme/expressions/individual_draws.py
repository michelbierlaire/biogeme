"""Function defining the draws per individuals, and not per observation.

Michel Bierlaire
Sun Nov 09 2025, 17:18:38
"""

from typing import Any

from .base_expressions import Expression
from .draws import Draws
from .visitor import ExpressionVisitor

_individual_visitor = ExpressionVisitor()
register_individual = _individual_visitor.register


@register_individual(Draws)
def individual_draws_handler(expr: Draws, context: dict[str, Any]) -> None:
    expr.set_draw_per_individual()


def individual_draws(expr: Expression) -> None:
    context = {}
    _individual_visitor.visit(expr, context)
