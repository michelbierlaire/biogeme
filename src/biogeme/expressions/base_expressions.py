"""Arithmetic expressions accepted by Biogeme: generic class

Michel Bierlaire
Tue Mar 25 17:00:02 2025
"""

from __future__ import annotations

import logging
from itertools import chain
from typing import NamedTuple, TYPE_CHECKING, TypeAlias

from biogeme.exceptions import BiogemeError, NotImplementedError

from .bayesian import PymcModelBuilderType
from .elementary_types import TypeOfElementaryExpression
from .jax_utils import JaxFunctionType
from .numeric_tools import is_numeric
from .validation import validate_expression_type

if TYPE_CHECKING:
    from .elementary_expressions import Elementary

logger = logging.getLogger(__name__)


class LogitTuple(NamedTuple):
    choice: Expression
    availabilities: dict[int, Expression]


class Expression:
    """This is the general arithmetic expression in biogeme.
    It serves as a base class for concrete expressions.
    """

    def __init__(self) -> None:
        """Constructor"""
        self.children = []  #: List of children expressions

        self.fixed_beta_values = None
        """values of the Beta that are not estimated
        """

        self._is_complex = False
        """ Flag identifying complex expressions 
        """

    def is_complex(self) -> bool:
        """Determine if the expression is complex.

        An expression is considered complex if its own _is_complex flag
        is set or if any of its children are complex.

        :return: True if the expression or any of its children is complex.
        """
        return self._is_complex or any(child.is_complex() for child in self.children)

    def __bool__(self) -> None:
        error_msg = f'Expression {str(self)} cannot be used in a boolean expression. Use & for "and" and | for "or"'
        raise BiogemeError(error_msg)

    def get_value(self) -> float:
        """Calculates the value of the expression if it is simple"""
        raise BiogemeError(
            f'Expression of type {self.get_class_name()} does not support direct value evaluation.'
        )

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates recursively a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        raise NotImplementedError(
            f"recursive_construct_pymc_model_builder not implemented for {type(self).__name__}"
        )

    def recursive_construct_jax_function(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        raise NotImplementedError(
            f"recursive_construct_pymc_model_builder not implemented for {type(self).__name__}"
        )

    def deep_flat_copy(self) -> Expression:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        """
        raise NotImplementedError(
            f"deep_flat_copy not implemented for {type(self).__name__}"
        )

    def __add__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added

        :return: self + other

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .plus import Plus

        return Plus(self, other)

    def __radd__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added
        :param other: expression to be added
        :type other: biogeme.expressions.Expression

        :return: other + self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        validate_expression_type(other)
        from .plus import Plus

        return Plus(other, self)

    def __sub__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for subtraction.

        :param other: expression to subtract
        :type other: biogeme.expressions.Expression

        :return: self - other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        validate_expression_type(other)
        from .minus import Minus

        return Minus(self, other)

    def __rsub__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for subtraction.

        :param other: expression to be subtracted
        :type other: biogeme.expressions.Expression

        :return: other - self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        validate_expression_type(other)
        from .minus import Minus

        return Minus(other, self)

    def __mul__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for multiplication.

        :param other: expression to be multiplied
        :type other: biogeme.expressions.Expression

        :return: self * other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        validate_expression_type(other)
        from .times import Times

        return Times(self, other)

    def __rmul__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for multiplication.

        :param other: expression to be multiplied
        :type other: biogeme.expressions.Expression

        :return: other * self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .times import Times

        return Times(other, self)

    def __truediv__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: self / other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .divide import Divide

        return Divide(self, other)

    def __rtruediv__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: other / self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .divide import Divide

        return Divide(other, self)

    def __neg__(self) -> Expression:
        """
        Operator overloading. Generate an expression for unary minus.

        :return: -self
        :rtype: biogeme.expressions.Expression
        """
        from .unary_minus import UnaryMinus

        return UnaryMinus(self)

    def __pow__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for power.

        :param other: expression for power
        :type other: biogeme.expressions.Expression

        :return: self ^ other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        validate_expression_type(other)
        if is_numeric(other):
            from .power_constant import PowerConstant

            return PowerConstant(child=self, exponent=float(other))

        from .numeric_expressions import Numeric

        if isinstance(other, Numeric):
            from .power_constant import PowerConstant

            return PowerConstant(self, other.get_value())

        if isinstance(other, Expression):
            from .power import Power

            return Power(self, other)

    def __rpow__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for power.

        :param other: expression for power
        :type other: biogeme.expressions.Expression

        :return: other ^ self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .power import Power

        return Power(other, self)

    def __and__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for logical and.

        :param other: expression for logical and
        :type other: biogeme.expressions.Expression

        :return: self and other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .logical_and import And

        return And(self, other)

    def __rand__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for logical and.

        :param other: expression for logical and
        :type other: biogeme.expressions.Expression

        :return: other and self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .logical_and import And

        return And(other, self)

    def __or__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for logical or.

        :param other: expression for logical or
        :type other: biogeme.expressions.Expression

        :return: self or other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .logical_or import Or

        return Or(self, other)

    def __ror__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for logical or.

        :param other: expression for logical or
        :type other: biogeme.expressions.Expression

        :return: other or self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .logical_or import Or

        return Or(other, self)

    def __eq__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for equality
        :type other: biogeme.expressions.Expression

        :return: self == other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .comparison_expressions import Equal

        return Equal(self, other)

    def __ne__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for difference
        :type other: biogeme.expressions.Expression

        :return: self != other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .comparison_expressions import NotEqual

        return NotEqual(self, other)

    def __le__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for less or equal
        :type other: biogeme.expressions.Expression

        :return: self <= other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .comparison_expressions import LessOrEqual

        return LessOrEqual(self, other)

    def __ge__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for greater or equal
        :type other: biogeme.expressions.Expression

        :return: self >= other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .comparison_expressions import GreaterOrEqual

        return GreaterOrEqual(self, other)

    def __lt__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for less than
        :type other: biogeme.expressions.Expression

        :return: self < other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .comparison_expressions import Less

        return Less(self, other)

    def __gt__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for greater than
        :type other: biogeme.expressions.Expression

        :return: self > other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        validate_expression_type(other)
        from .comparison_expressions import Greater

        return Greater(self, other)

    def logit_choice_avail(self) -> list[LogitTuple]:
        """Extract a dict with all elementary expressions of a specific type

        :return: returns a dict with the variables appearing in the
               expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        result: list[LogitTuple] = list(
            chain.from_iterable(e.logit_choice_avail() for e in self.children)
        )
        return result

    def get_elementary_expression(self, name: str) -> Elementary | None:
        """Return: an elementary expression from its name if it appears in the
        expression.

        :param name: name of the elementary expression.
        :type name: string

        :return: the expression if it exists. None otherwise.
        :rtype: biogeme.expressions.Expression
        """
        for e in self.get_children():
            if e.get_elementary_expression(name) is not None:
                return e.get_elementary_expression(name)
        return None

    def get_class_name(self) -> str:
        """
        Obtain the name of the top class of the expression structure

        :return: the name of the class
        :rtype: string
        """
        n = type(self).__name__
        return n

    def get_children(self) -> list[Expression]:
        """Retrieve the list of children

        :return: list of children
        :rtype: list(Expression)
        """
        return self.children

    def embed_expression(self, name: type) -> bool:
        """Check if an expression embeds a specific operator"""
        if isinstance(self, name):
            return True
        return any(child.embed_expression(name) for child in self.get_children())

    def set_specific_id(
        self, the_name, specific_id, the_type: TypeOfElementaryExpression
    ):
        """The elementary IDs identify the position of each element in the corresponding datab"""
        for child in self.get_children():
            child.set_specific_id(the_name, specific_id, the_type)

    def set_maximum_number_of_observations_per_individual(
        self, max_number: int
    ) -> None:
        for child in self.get_children():
            child.set_maximum_number_of_observations_per_individual(
                max_number=max_number
            )

    def change_init_values(self, betas: dict[str, float]):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """
        for child in self.get_children():
            child.change_init_values(betas=betas)

    def fix_betas(
        self,
        beta_values: dict[str, float],
        prefix: str | None = None,
        suffix: str | None = None,
    ):
        """Fix all the values of the Beta parameters appearing in the
        dictionary

        :param beta_values: dictionary containing the betas to be
            fixed (as key) and their value.
        :type beta_values: dict(str: float)

        :param prefix: if not None, the parameter is renamed, with a
            prefix defined by this argument.
        :type prefix: str

        :param suffix: if not None, the parameter is renamed, with a
            suffix defined by this argument.
        :type suffix: str

        """
        for child in self.get_children():
            child.fix_betas(beta_values=beta_values, prefix=prefix, suffix=suffix)

    def requires_draws(self):
        from .draws import Draws

        return self.embed_expression(Draws)


ExpressionOrNumeric: TypeAlias = Expression | float | int | bool
