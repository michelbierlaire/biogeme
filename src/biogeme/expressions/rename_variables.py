"""Function renaming variables inside an expression

Michel Bierlaire
Fri Jul 25 2025, 18:08:38
"""

from typing import NamedTuple

from .base_expressions import Expression
from .variable import Variable
from .visitor import ExpressionVisitor

_rename_visitor = ExpressionVisitor()
register_rename = _rename_visitor.register


class OldNewName(NamedTuple):
    old_name: str
    new_name: str


@register_rename(Variable)
def rename_variable_handler(expr, context):
    if expr.name == context['old_name']:
        expr.name = context['new_name']
        context['count'] += 1


def rename_all_variables(expr: Expression, renaming_list: list[OldNewName]) -> int:
    total_count = 0
    for renaming in renaming_list:
        context = {
            'old_name': renaming.old_name,
            'new_name': renaming.new_name,
            'count': 0,
        }
        _rename_visitor.visit(expr, context)
        total_count += context['count']
    return total_count
