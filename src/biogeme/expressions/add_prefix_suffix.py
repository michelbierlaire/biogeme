"""Function adding a suffix to variables.

Michel Bierlaire
Fri May 02 2025, 17:53:00
"""

from .base_expressions import Expression
from .variable import Variable
from .visitor import ExpressionVisitor

_prefix_suffix_visitor = ExpressionVisitor()
register_prefix_suffix = _prefix_suffix_visitor.register


@register_prefix_suffix(Variable)
def prefix_suffix_variable_name_handler(expr, context):
    expr_id = id(expr)
    if expr_id in context["visited_ids"]:
        return

    context["visited_ids"].add(expr_id)
    old_name = expr.name
    new_name = f"{context['prefix']}{old_name}{context['suffix']}"
    expr.name = new_name
    context["count"] += 1


def add_prefix_suffix_to_all_variables(
    expr: Expression, prefix: str, suffix: str
) -> int:
    context = {
        "prefix": prefix,
        "suffix": suffix,
        "count": 0,
        "visited_ids": set(),
    }
    _prefix_suffix_visitor.visit(expr, context)
    return context["count"]
