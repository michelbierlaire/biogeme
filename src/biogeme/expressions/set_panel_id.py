"""Function defining the panel ID.

Michel Bierlaire
Sun Nov 09 2025, 17:38:08
"""

from typing import Any

from .base_expressions import Expression
from .panel_log_likelihood import PanelLogLikelihood
from .visitor import ExpressionVisitor

_panel_id_visitor = ExpressionVisitor()
register_panel_id = _panel_id_visitor.register

PANEL_ID = 'panel_id'


@register_panel_id(PanelLogLikelihood)
def panel_id_handler(expr: PanelLogLikelihood, context: dict[str, Any]) -> None:
    expr.panel_id = context[PANEL_ID]
    return None


def set_panel_id(expr: Expression, panel_id: str) -> None:
    context = {PANEL_ID: panel_id}
    _panel_id_visitor.visit(expr, context)
