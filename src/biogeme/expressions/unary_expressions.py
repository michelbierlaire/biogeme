""" Arithmetic expressions accepted by Biogeme: unary operators

:author: Michel Bierlaire
:date: Sat Sep  9 15:51:53 2023
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

from . import validate_and_convert
from .base_expressions import Expression
from ..deprecated import deprecated


if TYPE_CHECKING:
    from biogeme.database import Database
    from . import ExpressionOrNumeric

logger = logging.getLogger(__name__)


class UnaryOperator(Expression):
    """
    Base class for arithmetic expressions that are unary operators.

    Such an expression is the result of the modification of another
    expressions, typically changing its sign.
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        Expression.__init__(self)
        self.child = validate_and_convert(child)
        self.children.append(self.child)


class UnaryMinus(UnaryOperator):
    """
    Unary minus expression

    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'(-{self.child})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return -self.child.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        pass


class MonteCarlo(UnaryOperator):
    """
    Monte Carlo integration
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'MonteCarlo({self.child})'

    def check_draws(self) -> set[str]:
        """List of draws defined outside of 'MonteCarlo'

        :return: List of names of variables
        :rtype: list(str)
        """
        return set()

    def audit(self, database: Database = None) -> tuple[list[str], list[str]]:
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)

        """
        list_of_errors, list_of_warnings = self.child.audit(database)
        if database is None:
            if self.child.embed_expression('PanelLikelihoodTrajectory'):
                the_warning = (
                    'The formula contains a PanelLikelihoodTrajectory '
                    'expression, and no database is given'
                )
                list_of_warnings.append(the_warning)
        else:
            if database.is_panel() and not self.child.embed_expression(
                'PanelLikelihoodTrajectory'
            ):
                the_error = (
                    f'As the database is panel, the argument '
                    f'of MonteCarlo must contain a'
                    f' PanelLikelihoodTrajectory: {self}'
                )
                list_of_errors.append(the_error)

        if not self.child.embed_expression('bioDraws'):
            the_error = (
                f'The argument of MonteCarlo must contain a' f' bioDraws: {self}'
            )
            list_of_errors.append(the_error)
        if self.child.embed_expression('MonteCarlo'):
            the_error = (
                f'It is not possible to include a MonteCarlo '
                f'statement in another one: {self}'
            )
            list_of_errors.append(the_error)
        return list_of_errors, list_of_warnings


class bioNormalCdf(UnaryOperator):
    """
    Cumulative Distribution Function of a normal random variable
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'bioNormalCdf({self.child})'


class PanelLikelihoodTrajectory(UnaryOperator):
    """
    Likelihood of a sequences of observations for the same individual
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'PanelLikelihoodTrajectory({self.child})'

    def check_panel_trajectory(self) -> set[str]:
        """List of variables defined outside of 'PanelLikelihoodTrajectory'

        :return: List of names of variables
        :rtype: list(str)
        """
        return set()

    def count_panel_trajectory_expressions(self) -> int:
        """Count the number of times the PanelLikelihoodTrajectory
        is used in the formula.
        """
        return 1 + self.child.count_panel_trajectory_expressions()

    def audit(self, database: Database = None) -> tuple[list[str], list[str]]:
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)

        """
        list_of_errors, list_of_warnings = self.child.audit(database)
        if not database.is_panel():
            the_error = (
                f'Expression PanelLikelihoodTrajectory can '
                f'only be used with panel data. Use the statement '
                f'database.panel("IndividualId") to declare the '
                f'panel structure of the data: {self}'
            )
            list_of_errors.append(the_error)
        return list_of_errors, list_of_warnings


class exp(UnaryOperator):
    """
    exponential expression
    """

    def __init__(self, child: ExpressionOrNumeric) -> None:
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'exp({self.child})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.exp(self.child.get_value())

    @deprecated(get_value)
    def getValue(self) -> float:
        pass


class sin(UnaryOperator):
    """
    sine expression
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'sin({self.child})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.sin(self.child.get_value())

    @deprecated(get_value)
    def getValue(self) -> float:
        pass


class cos(UnaryOperator):
    """
    cosine expression
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'cos({self.child})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.cos(self.child.get_value())

    @deprecated(get_value)
    def getValue(self) -> float:
        pass


class log(UnaryOperator):
    """
    logarithm expression
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'log({self.child})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.log(self.child.get_value())

    @deprecated(get_value)
    def getValue(self) -> float:
        pass


class logzero(UnaryOperator):
    """
    logarithm expression. Returns zero if the argument is zero.
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self) -> str:
        return f'logzero({self.child})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        v = self.child.get_value()
        return 0 if v == 0 else np.log(v)

    @deprecated(get_value)
    def getValue(self) -> float:
        pass


class PowerConstant(UnaryOperator):
    """
    Raise the argument to a constant power.
    """

    def __init__(self, child: ExpressionOrNumeric, exponent: float):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)
        self.exponent: float = exponent
        self.integer_exponent: int | None = (
            int(exponent) if exponent.is_integer() else None
        )

    def __str__(self) -> str:
        return f'{self.child}**{self.exponent}'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        v = self.child.get_value()
        if v == 0:
            return 0.0
        if v > 0:
            return v**self.exponent
        if self.integer_exponent is not None:
            return v**self.integer_exponent
        if v < 0:
            error_msg = f'Cannot calculate {v}**{self.exponent}'
            raise (error_msg)

        return 0 if v == 0 else np.log(v)

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of the child expression,
            2. the name of the expression between < >
            3. the id of the expression between { }, preceded by a comma
            4. the id of the children, preceded by a comma
            5. the index of the randon variable, preceded by a comma

        Consider the following expression:

        .. math:: 2 \\beta_1  V_1 -
             \\frac{\\exp(-\\beta_2 V_2) }
             { \\beta_3  (\\beta_2 \\geq \\beta_1)}.

        It is defined as::

            2 * beta1 * Variable1 - expressions.exp(-beta2*Variable2) /
                 (beta3 * (beta2 >= beta1))

        And its signature is::

            [b'<Numeric>{4780527008},2',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<Times>{4780526952}(2),4780527008,4780277152',
             b'<Variable>{4511837152}"Variable1",5,2',
             b'<Times>{4780527064}(2),4780526952,4511837152',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<UnaryMinus>{4780527120}(1),4780277656',
             b'<Variable>{4511837712}"Variable2",6,3',
             b'<Times>{4780527176}(2),4780527120,4511837712',
             b'<exp>{4780527232}(1),4780527176',
             b'<Beta>{4780277264}"beta3"[1],2,0',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<GreaterOrEqual>{4780527288}(2),4780277656,4780277152',
             b'<Times>{4780527344}(2),4780277264,4780527288',
             b'<Divide>{4780527400}(2),4780527232,4780527344',
             b'<Minus>{4780527456}(2),4780527064,4780527400']

        :return: list of the signatures of an expression and its children.
        :rtype: list(string)
        """
        list_of_signatures = []
        list_of_signatures += self.child.get_signature()
        signature = f'<{self.get_class_name()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f',{self.child.get_id()}'
        signature += f',{self.exponent}'
        list_of_signatures += [signature.encode()]
        return list_of_signatures


class Derive(UnaryOperator):
    """
    Derivative with respect to an elementary expression
    """

    def __init__(self, child: ExpressionOrNumeric, name: str):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)
        # Name of the elementary expression by which the derivative is taken
        self.elementaryName = name

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of the child expression,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the id of the child, preceded by a comma.

        Consider the following expression:

        .. math:: 2 \\beta_1  V_1 -
           \\frac{\\exp(-\\beta_2 V_2) }{ \\beta_3  (\\beta_2 \\geq \\beta_1)}.

        It is defined as::

            2 * beta1 * Variable1 - expressions.exp(-beta2*Variable2) /
                (beta3 * (beta2 >= beta1))

        And its signature is::

            [b'<Numeric>{4780527008},2',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<Times>{4780526952}(2),4780527008,4780277152',
             b'<Variable>{4511837152}"Variable1",5,2',
             b'<Times>{4780527064}(2),4780526952,4511837152',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<UnaryMinus>{4780527120}(1),4780277656',
             b'<Variable>{4511837712}"Variable2",6,3',
             b'<Times>{4780527176}(2),4780527120,4511837712',
             b'<exp>{4780527232}(1),4780527176',
             b'<Beta>{4780277264}"beta3"[1],2,0',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<GreaterOrEqual>{4780527288}(2),4780277656,4780277152',
             b'<Times>{4780527344}(2),4780277264,4780527288',
             b'<Divide>{4780527400}(2),4780527232,4780527344',
             b'<Minus>{4780527456}(2),4780527064,4780527400']

        :return: list of the signatures of an expression and its children.
        :rtype: list(string)
        """
        elementary_index = self.id_manager.elementary_expressions.indices[
            self.elementaryName
        ]
        list_of_signatures = []
        list_of_signatures += self.child.get_signature()
        my_signature = f'<{self.get_class_name()}>'
        my_signature += f'{{{self.get_id()}}}'
        my_signature += f',{self.child.get_id()}'
        my_signature += f',{elementary_index}'
        list_of_signatures += [my_signature.encode()]
        return list_of_signatures

    def __str__(self) -> str:
        return 'Derive({self.child}, "{self.elementName}")'


class Integrate(UnaryOperator):
    """
    Numerical integration
    """

    def __init__(self, child: ExpressionOrNumeric, name: str):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        :param name: name of the random variable for the integration.
        :type name: string
        """
        UnaryOperator.__init__(self, child)
        self.randomVariableName = name

    def check_rv(self) -> set[str]:
        """List of random variables defined outside of 'Integrate'

        :return: List of names of variables
        :rtype: list(str)
        """
        return set()

    def audit(self, database: Database = None) -> tuple[list[str], list[str]]:
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)

        """
        list_of_errors, list_of_warnings = self.child.audit(database)
        if not self.child.embed_expression('RandomVariable'):
            the_error = (
                f'The argument of Integrate must contain a ' f'RandomVariable: {self}'
            )
            list_of_errors.append(the_error)
        return list_of_errors, list_of_warnings

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of the child expression,
            2. the name of the expression between < >
            3. the id of the expression between { }, preceeded by a comma
            4. the id of the children, preceeded by a comma
            5. the index of the randon variable, preceeded by a comma

        Consider the following expression:

        .. math:: 2 \\beta_1  V_1 -
             \\frac{\\exp(-\\beta_2 V_2) }
             { \\beta_3  (\\beta_2 \\geq \\beta_1)}.

        It is defined as::

            2 * beta1 * Variable1 - expressions.exp(-beta2*Variable2) /
                 (beta3 * (beta2 >= beta1))

        And its signature is::

            [b'<Numeric>{4780527008},2',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<Times>{4780526952}(2),4780527008,4780277152',
             b'<Variable>{4511837152}"Variable1",5,2',
             b'<Times>{4780527064}(2),4780526952,4511837152',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<UnaryMinus>{4780527120}(1),4780277656',
             b'<Variable>{4511837712}"Variable2",6,3',
             b'<Times>{4780527176}(2),4780527120,4511837712',
             b'<exp>{4780527232}(1),4780527176',
             b'<Beta>{4780277264}"beta3"[1],2,0',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<GreaterOrEqual>{4780527288}(2),4780277656,4780277152',
             b'<Times>{4780527344}(2),4780277264,4780527288',
             b'<Divide>{4780527400}(2),4780527232,4780527344',
             b'<Minus>{4780527456}(2),4780527064,4780527400']

        :return: list of the signatures of an expression and its children.
        :rtype: list(string)
        """
        random_variable_index = self.id_manager.random_variables.indices[
            self.randomVariableName
        ]
        list_of_signatures = []
        list_of_signatures += self.child.get_signature()
        my_signature = f'<{self.get_class_name()}>'
        my_signature += f'{{{self.get_id()}}}'
        my_signature += f',{self.child.get_id()}'
        my_signature += f',{random_variable_index}'
        list_of_signatures += [my_signature.encode()]
        return list_of_signatures

    def __str__(self) -> str:
        return f'Integrate({self.child}, "{self.randomVariableName}")'


class BelongsTo(UnaryOperator):
    """
    Check if a value belongs to a set
    """

    def __init__(self, child: ExpressionOrNumeric, the_set: set[float]):
        """Constructor

        :param child: arithmetic expression
        :type child: biogeme.expressions.Expression
        :param the_set: set of values
        :type the_set: set(float)
        """
        UnaryOperator.__init__(self, child)
        self.the_set = the_set

    def audit(self, database: Database = None) -> tuple[list[str], list[str]]:
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)

        """
        list_of_errors, list_of_warnings = self.child.audit(database)
        if not all(float(x).is_integer() for x in self.the_set):
            the_warning = (
                f'The set of numbers used in the expression "BelongsTo" contains '
                f'numbers that are not integer. If it is the intended use, ignore '
                f'this warning: {self.the_set}.'
            )
            list_of_warnings.append(the_warning)
        return list_of_errors, list_of_warnings

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of the child expression,
            2. the name of the expression between < >
            3. the id of the expression between { }, preceded by a comma
            4. the id of the children, preceded by a comma
            5. the index of the randon variable, preceded by a comma

        Consider the following expression:

        .. math:: 2 \\beta_1  V_1 -
             \\frac{\\exp(-\\beta_2 V_2) }
             { \\beta_3  (\\beta_2 \\geq \\beta_1)}.

        It is defined as::

            2 * beta1 * Variable1 - expressions.exp(-beta2*Variable2) /
                 (beta3 * (beta2 >= beta1))

        And its signature is::

            [b'<Numeric>{4780527008},2',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<Times>{4780526952}(2),4780527008,4780277152',
             b'<Variable>{4511837152}"Variable1",5,2',
             b'<Times>{4780527064}(2),4780526952,4511837152',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<UnaryMinus>{4780527120}(1),4780277656',
             b'<Variable>{4511837712}"Variable2",6,3',
             b'<Times>{4780527176}(2),4780527120,4511837712',
             b'<exp>{4780527232}(1),4780527176',
             b'<Beta>{4780277264}"beta3"[1],2,0',
             b'<Beta>{4780277656}"beta2"[0],1,1',
             b'<Beta>{4780277152}"beta1"[0],0,0',
             b'<GreaterOrEqual>{4780527288}(2),4780277656,4780277152',
             b'<Times>{4780527344}(2),4780277264,4780527288',
             b'<Divide>{4780527400}(2),4780527232,4780527344',
             b'<Minus>{4780527456}(2),4780527064,4780527400']

        :return: list of the signatures of an expression and its children.
        :rtype: list(string)
        """
        list_of_signatures = []
        list_of_signatures += self.child.get_signature()
        signature = f'<{self.get_class_name()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f'({len(self.the_set)})'
        signature += f',{self.child.get_id()}'
        for elem in self.the_set:
            signature += f',{elem}'
        list_of_signatures += [signature.encode()]
        return list_of_signatures

    def __str__(self) -> str:
        return f'BelongsTo({self.child}, "{self.the_set}")'
