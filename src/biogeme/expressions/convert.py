""" Convert expressions to float and vice versa

Michel Bierlaire
Thu Apr 11 15:31:15 2024
"""

from __future__ import annotations

from biogeme.exceptions import BiogemeError
from biogeme.expressions import ExpressionOrNumeric, Expression, Numeric, is_numeric


def validate_and_convert(expression: ExpressionOrNumeric) -> Expression:
    """Validates the expression and returns the converted expression if necessary."""
    if isinstance(expression, bool):
        return Numeric(1) if expression else Numeric(0)
    if is_numeric(expression):
        return Numeric(expression)
    if not isinstance(expression, Expression):
        raise TypeError(f'This is not a valid expression: {expression}')
    return expression


def expression_to_value(
    expression: ExpressionOrNumeric, betas: dict[str, float] | None = None
) -> float:
    """
    Convert to float, if possible
    :param expression: expression to be converted
    :return: numerical value
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
        error_msg = f'Expression too complex to be associated with a numeric value: {e}'
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
    the_dict: dict[int, ExpressionOrNumeric]
) -> dict[int, Expression]:
    """If the dictionary contains float, they are transformed into a
    numerical expression."""

    return {key: validate_and_convert(value) for key, value in the_dict.items()}
