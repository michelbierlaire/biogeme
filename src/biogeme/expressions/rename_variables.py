"""Function renaming variables inside an expression

Michel Bierlaire
Fri May 02 2025, 13:51:23
"""

from .base_expressions import Expression
from .variable import Variable
from .visitor import ExpressionVisitor

_rename_visitor = ExpressionVisitor()
register_rename = _rename_visitor.register


@register_rename(Variable)
def rename_variable_handler(expr, context):
    if expr.name == context['old_name']:
        expr.name = context['new_name']
        context['count'] += 1


def rename_all_variables(expr: Expression, old_name: str, new_name: str) -> int:
    context = {'old_name': old_name, 'new_name': new_name, 'count': 0}
    _rename_visitor.visit(expr, context)
    return context['count']
