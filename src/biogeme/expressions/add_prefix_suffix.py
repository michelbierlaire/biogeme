"""Function adding a suffix to variables

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
    expr.name = f"{context['prefix']}{expr.name}{context['suffix']}"
    context['count'] += 1


def add_prefix_suffix_to_all_variables(
    expr: Expression, prefix: str, suffix: str
) -> int:
    context = {'prefix': prefix, 'suffix': suffix, 'count': 0}
    _prefix_suffix_visitor.visit(expr, context)
    return context['count']
