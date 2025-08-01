"""Functions to validate the arithmetic expressions

Michel Bierlaire
Tue Mar 25 17:17:11 2025
"""

from typing import Any


from .numeric_tools import is_numeric
from ..exceptions import BiogemeError


def validate_expression_type(expression: Any) -> None:
    """
    :param expression: expression to validate
    """
    from .base_expressions import Expression

    if not (is_numeric(expression) or isinstance(expression, Expression)):
        error_msg = f'Invalid expression: {str(expression)}'
        raise BiogemeError(error_msg)
