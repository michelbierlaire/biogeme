"""Convert expressions to float and vice versa

Michel Bierlaire
Tue Mar 25 18:43:36 2025
"""

from __future__ import annotations

from biogeme.exceptions import BiogemeError
from .base_expressions import ExpressionOrNumeric, Expression
from .numeric_expressions import Numeric
from .numeric_tools import is_numeric


def validate_and_convert(expression: ExpressionOrNumeric) -> Expression:
    """Validates the expression and returns the converted expression if necessary."""
    if isinstance(expression, bool):
        return Numeric(1) if expression else Numeric(0)
    if is_numeric(expression):
        return Numeric(expression)
    if not isinstance(expression, Expression):
        raise TypeError(
            f'This is not a valid expression: {expression}. It is of type {type(expression)}'
        )
    return expression


def expression_to_value(
    expression: ExpressionOrNumeric, betas: dict[str, float] | None = None
) -> float:
    """
    Convert an expression to a float value, if possible.

    :param expression: The expression to convert. Can be a boolean, a numeric value,
                       or an instance of Expression.
    :param betas: Optional dictionary of beta values used to initialize the expression,
                  if applicable.
    :return: The numerical value of the expression as a float.
    :raises TypeError: If the input is not a valid expression type.
    :raises BiogemeError: If the expression cannot be evaluated to a numeric value.
    """
    if isinstance(expression, bool) or is_numeric(expression):
        return float(expression)
    if not isinstance(expression, Expression):
        raise TypeError(f'This is not a valid expression: {expression}')
    if betas is not None:
        expression.change_init_values(betas=betas)
    try:
        value = expression.get_value()
    except BiogemeError as e:
        error_msg = f'Expression {expression} too complex to be associated with a numeric value: {e}'
        raise BiogemeError(error_msg) from e
    return value


def get_dict_values(
    the_dict: dict[int, ExpressionOrNumeric], betas: dict[str, float] | None = None
) -> dict[int, float]:
    """If the dictionary contains Expressions, they are transformed into a
    numerical expression."""

    return {
        key: expression_to_value(expression=expr, betas=betas)
        for key, expr in the_dict.items()
    }


def get_dict_expressions(
    the_dict: dict[int, ExpressionOrNumeric],
) -> dict[int, Expression]:
    """If the dictionary contains float, they are transformed into a
    numerical expression."""

    return {key: validate_and_convert(value) for key, value in the_dict.items()}
