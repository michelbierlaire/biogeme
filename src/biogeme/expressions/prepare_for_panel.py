from typing import Literal, NamedTuple

from biogeme.expressions import DistributedParameter, Draws

from .base_expressions import Expression
from .visitor import ExpressionVisitor

_panel_column_visitor = ExpressionVisitor()
register_panel_column = _panel_column_visitor.register

_set_draw_dimension_visitor = ExpressionVisitor()
register_set_draw_dimension = _set_draw_dimension_visitor.register


class PanelColumnSetting(NamedTuple):
    panel_column: str


@register_panel_column(DistributedParameter)
def panel_column_handler(expr: DistributedParameter, context: dict) -> None:
    # Set / overwrite the panel_column attribute
    expr.panel_column = context["panel_column"]
    context["count"] += 1


@register_set_draw_dimension(Draws)
def set_draw_dimension_handler(expr: Draws, context: dict) -> None:
    mode = context["mode"]

    if mode == "observation":
        expr.set_draw_per_observation()
    elif mode == "individual":
        expr.set_draw_per_individual()
    else:
        raise ValueError(f"Unknown draw dimension mode: {mode!r}")

    context["count"] += 1


def set_panel_column_on_distributed_parameters(
    expr: Expression,
    panel_column: str,
) -> int:
    """
    Sets `panel_column` on all DistributedParameter nodes in `expr`.

    :param expr: Root expression.
    :param panel_column: Name of the panel id column in the dataframe.
    :return: Number of DistributedParameter nodes updated.
    """
    context = {"panel_column": panel_column, "count": 0}
    _panel_column_visitor.visit(expr, context)
    return context["count"]


def set_draw_dimension_for_all_draws(
    expr,
    mode: Literal["observation", "individual"],
) -> int:
    """
    Sets the draw dimension for all Draws expressions inside `expr`.

    :param expr: Root expression of the model.
    :param mode: {"observation", "individual"} Whether draws should be generated per observation or per individual.
    :return:  Number of Draws expressions updated.
    """
    context = {"mode": mode, "count": 0}
    _set_draw_dimension_visitor.visit(expr, context)
    return context["count"]


def prepare_for_panel(expr: Expression, panel_column: str) -> None:
    set_panel_column_on_distributed_parameters(expr=expr, panel_column=panel_column)
    set_draw_dimension_for_all_draws(expr=expr, mode="individual")
