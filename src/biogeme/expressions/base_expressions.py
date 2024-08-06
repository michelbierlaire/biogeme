""" Arithmetic expressions accepted by Biogeme: generic class

:author: Michel Bierlaire
:date: Sat Sep  9 15:25:07 2023
"""

from __future__ import annotations

import logging
from itertools import chain
from typing import TYPE_CHECKING, Callable, Iterable

from biogeme.configuration import Configuration
from biogeme.controller import CentralController, Controller
from biogeme.deprecated import deprecated, deprecated_parameters
from biogeme.function_output import (
    BiogemeFunctionOutput,
    BiogemeDisaggregateFunctionOutput,
    BiogemeFunctionOutputSmartOutputProxy,
    FunctionOutput,
    NamedBiogemeDisaggregateFunctionOutput,
    NamedBiogemeFunctionOutput,
    BiogemeDisaggregateFunctionOutputSmartOutputProxy,
    NamedFunctionOutput,
)
from .catalog_iterator import SelectedExpressionsIterator

if TYPE_CHECKING:
    from biogeme.database import Database
    from biogeme.catalog import Catalog
    from .elementary_expressions import Elementary
    from . import MultipleExpression, Numeric

import numpy as np
from biogeme.exceptions import BiogemeError, NotImplementedError
from biogeme_optimization.function import FunctionToMinimize, FunctionData
from .idmanager import IdManager
from .numeric_tools import is_numeric
from .elementary_types import TypeOfElementaryExpression
from .calculator import calculate_function_and_derivatives

logger = logging.getLogger(__name__)


class Expression:
    """This is the general arithmetic expression in biogeme.
    It serves as a base class for concrete expressions.
    """

    def __init__(self) -> None:
        """Constructor"""

        self.children = []  #: List of children expressions

        self.id_manager = None  #: in charge of the IDs
        self.keep_id_manager = None  #: a copy of the ID manager

        self.fixedBetaValues = None
        """values of the Beta that are not estimated
        """

        self.numberOfDraws = None
        """number of draws for Monte Carlo integration
        """

        self._row = None
        """Row of the database where the values of the variables are found
        """

        self.missingData = 99999
        """ Value interpreted as missing data
        """

        self.central_controller = None
        """ Central controller for the multiple expressions
        """

    def __bool__(self):
        error_msg = f'Expression {str(self)} cannot be used in a boolean expression. Use & for "and" and | for "or"'
        raise BiogemeError(error_msg)

    def __iter__(self) -> SelectedExpressionsIterator:
        the_set = self.set_of_configurations()
        return SelectedExpressionsIterator(self, the_set)

    def check_panel_trajectory(self) -> set[str]:
        """Set of variables defined outside of 'PanelLikelihoodTrajectory'

        :return: List of names of variables
        """
        check_children = set(
            chain.from_iterable(
                [e.check_panel_trajectory() for e in self.get_children()]
            )
        )
        return check_children

    def check_draws(self) -> set[str]:
        """Set of draws defined outside of 'MonteCarlo'

        :return: List of names of variables
        """
        check_children = set(
            chain.from_iterable([e.check_draws() for e in self.get_children()])
        )
        return check_children

    def check_rv(self) -> set[str]:
        """Set of random variables defined outside of 'Integrate'

        :return: List of names of variables
        """
        check_children = set(
            chain.from_iterable([e.check_rv() for e in self.get_children()])
        )
        return check_children

    def get_status_id_manager(self) -> tuple[set[str], set[str]]:
        """Check the elementary expressions that are associated with
        an ID manager.

        :return: two sets of elementary expressions, those with and
            without an ID manager.
        """
        with_id = set()
        without_id = set()
        for e in self.get_children():
            yes, no = e.get_status_id_manager()
            with_id.update(yes)
            without_id.update(no)
        return with_id, without_id

    @deprecated(new_func=get_status_id_manager)
    def getStatusIdManager(self) -> tuple[set[str], set[str]]:
        """Kept for backward compatibility"""
        pass

    @deprecated_parameters(obsolete_params={'numberOfDraws': 'number_of_draws'})
    def prepare(self, database: "Database", number_of_draws: int) -> None:

        """Prepare the expression to be evaluated

        :param database: Biogeme database

        :param number_of_draws: number of draws for Monte-Carlo integration
        """
        # First, we reset the IDs, if any
        self.set_id_manager(None)
        # Second, we calculate a new set of IDs.
        id_manager = IdManager([self], database, number_of_draws)
        self.set_id_manager(id_manager)

    def set_id_manager(self, id_manager: IdManager | None) -> None:
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may nee to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """
        self.id_manager = id_manager
        for e in self.get_children():
            e.set_id_manager(id_manager)

    @deprecated(new_func=set_id_manager)
    def setIdManager(self, id_manager: IdManager | None) -> None:
        """Kept for backward compatibility"""
        pass

    def __repr__(self) -> str:
        """built-in function used to compute the 'official' string reputation
        of an object

        :return: description of the expression

        """
        return self.__str__()

    def __add__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added

        :return: self + other

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during addition to {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Plus

        return Plus(self, other)

    def __radd__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added
        :type other: biogeme.expressions.Expression

        :return: other + self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during addition to {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Plus

        return Plus(other, self)

    def __sub__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for substraction.

        :param other: expression to substract
        :type other: biogeme.expressions.Expression

        :return: self - other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during substraction to {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Minus

        return Minus(self, other)

    def __rsub__(self, other: ExpressionOrNumeric) -> Expression:
        """
        Operator overloading. Generate an expression for substraction.

        :param other: expression to be substracted
        :type other: biogeme.expressions.Expression

        :return: other - self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during substraction of {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Minus

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = (
                f"Invalid expression during multiplication " f"to {self}: [{other}]"
            )
            raise BiogemeError(error_msg)
        from .binary_expressions import Times

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = (
                f"Invalid expression during multiplication " f"to {self}: [{other}]"
            )
            raise BiogemeError(error_msg)
        from .binary_expressions import Times

        return Times(other, self)

    def __div__(self, other: ExpressionOrNumeric) -> Expression:
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
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division of {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Divide

        return Divide(self, other)

    def __rdiv__(self, other: ExpressionOrNumeric) -> Expression:
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
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division by {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Divide

        return Divide(other, self)

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division of {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Divide

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division by {self}: [{other}]"
            raise BiogemeError(error_msg)
        from .binary_expressions import Divide

        return Divide(other, self)

    def __neg__(self) -> Expression:
        """
        Operator overloading. Generate an expression for unary minus.

        :return: -self
        :rtype: biogeme.expressions.Expression
        """
        from .unary_expressions import UnaryMinus

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
        if is_numeric(other):
            from .unary_expressions import PowerConstant

            return PowerConstant(child=self, exponent=float(other))

        from . import Numeric

        if isinstance(other, Numeric):
            from .unary_expressions import PowerConstant

            return PowerConstant(self, other.get_value())

        if isinstance(other, Expression):
            from .binary_expressions import Power

            return Power(self, other)

        raise BiogemeError(f"This is not a valid expression: {other}")

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import Power

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import And

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import And

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import Or

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import Or

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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
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
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise BiogemeError(f"This is not a valid expression: {other}")
        from .comparison_expressions import Greater

        return Greater(self, other)

    @deprecated_parameters(obsolete_params={'numberOfDraws': 'number_of_draws'})
    def create_function(
        self,
        database: Database | None = None,
        number_of_draws: int = 1000,
        gradient: bool = True,
        hessian: bool = True,
        bhhh: bool = False,
    ) -> Callable[
        [np.ndarray],
        NamedBiogemeFunctionOutput,
    ]:
        """Create a function based on the expression. The function takes as
        argument an array for the free parameters, and return the
        value of the function, the gradient, the hessian and the BHHH. The
        calculation of the derivatives is optional.

        :param database: database. If no database is provided, the
            expression must not contain any variable.
        :type database:  biogeme.database.Database

        :param number_of_draws: number of draws if needed by Monte-Carlo
            integration.
        :type number_of_draws: int

        :param gradient: if True, the gradient is calculated.
        :type gradient: bool

        :param hessian: if True, the hessian is calculated.
        :type hessian: bool

        :param bhhh: if True, the BHHH matrix is calculated.
        :type bhhh: bool

        :return: the function. It will return, in that order, the
            value of the function, the gradient, the hessian and the
            BHHH matrix. Only requested quantities will be
            returned. For instance, if the gradient and the BHHH
            matrix are requested, and not the hessian, the tuple that
            is returned is f, g, bhhh.

        :rtype: fct(np.array)

        :raise BiogemeError: if gradient is False and hessian or BHHH is True.

        """
        if (hessian or bhhh) and not gradient:
            raise BiogemeError(
                "If the hessian or BHHH is calculated, so is the gradient. "
                "The provided parameters are inconsistent."
            )

        with_id, without_id = self.get_status_id_manager()
        if len(without_id) > 0:
            if len(with_id) > 0:
                error_msg = (
                    f"IDs are defined for some expressions "
                    f"[{with_id}] but not for some [{without_id}]"
                )
                raise BiogemeError(error_msg)
            self.set_id_manager(IdManager([self], database, number_of_draws))

        def my_function(
            x: np.ndarray,
        ) -> float | NamedBiogemeFunctionOutput:
            """Wrapper presenting the expression and its derivatives as a function"""
            if isinstance(x, (float, int, np.float64)):
                x = [float(x)]

            if len(x) != len(self.id_manager.free_betas_values):
                the_error_msg = (
                    f"Function is expecting an array of length "
                    f"{len(self.id_manager.free_betas_values)}, not {len(x)}"
                )
                raise BiogemeError(the_error_msg)

            self.id_manager.free_betas_values = x
            return self.get_value_and_derivatives(
                database=database,
                number_of_draws=number_of_draws,
                gradient=gradient,
                hessian=hessian,
                bhhh=bhhh,
                aggregation=True,
                prepare_ids=False,
                named_results=True,
            )

        return my_function

    @deprecated(create_function)
    def createFunction(
        self,
        database: Database | None = None,
        number_of_draws: int = 1000,
        gradient: bool = True,
        hessian: bool = True,
        bhhh: bool = False,
    ) -> Callable[
        [np.ndarray],
        float | BiogemeFunctionOutput,
    ]:
        """Kept for backward compatibility"""
        pass

    @deprecated_parameters(obsolete_params={'numberOfDraws': 'number_of_draws'})
    def create_objective_function(
        self,
        database: Database | None = None,
        number_of_draws: int = 1000,
        gradient: bool = True,
        hessian: bool = True,
        bhhh: bool = False,
    ) -> FunctionToMinimize:
        """Create a function based on the expression that complies
        with the interface of the biogeme_optimization model. The
        function takes as argument an array for the free parameters,
        and return the value of the function, the gradient, the
        hessian and the BHHH. The calculation of the derivatives is
        optional.

        :param database: database. If no database is provided, the
            expression must not contain any variable.
        :type database:  biogeme.database.Database

        :param number_of_draws: number of draws if needed by Monte-Carlo
            integration.
        :type number_of_draws: int

        :param gradient: if True, the gradient is calculated.
        :type gradient: bool

        :param hessian: if True, the hessian is calculated.
        :type hessian: bool

        :param bhhh: if True, the BHHH matrix is calculated.
        :type bhhh: bool

        :return: the function object.
        :rtype: FunctionToMinimize


        :raise BiogemeError: if gradient is False and hessian or BHHH is True.
        """
        try:
            expression_function = self.create_function(
                database,
                number_of_draws,
                gradient=False,
                hessian=False,
                bhhh=False,
            )
        except BiogemeError as e:
            raise e

        expression_function_gradient = self.create_function(
            database,
            number_of_draws,
            gradient=True,
            hessian=False,
            bhhh=False,
        )
        expression_function_gradient_hessian = self.create_function(
            database,
            number_of_draws,
            gradient=True,
            hessian=True,
            bhhh=False,
        )

        class Function(FunctionToMinimize):
            """Class encapsulating the expression into a FunctionToMinimize"""

            def __init__(
                self, epsilon: float | None = None, steptol: float | None = None
            ):
                super().__init__(epsilon, steptol)
                self.idmanager = None

            def _f(self) -> float:

                the_function_output: FunctionOutput = expression_function(
                    self.x
                ).function_output
                return the_function_output.function

            def _f_g(self) -> FunctionData:
                the_function_output: FunctionOutput = expression_function_gradient(
                    self.x
                ).function_output
                return FunctionData(
                    function=the_function_output.function,
                    gradient=the_function_output.gradient,
                    hessian=None,
                )

            def _f_g_h(self) -> FunctionData:
                the_function_output: FunctionOutput = (
                    expression_function_gradient_hessian(self.x)
                ).function_output
                return FunctionData(
                    function=the_function_output.function,
                    gradient=the_function_output.gradient,
                    hessian=the_function_output.hessian,
                )

            def dimension(self) -> int:
                return self.idmanager.number_of_free_betas

        return Function()

    def get_value(self) -> float:
        """Abstract method"""
        raise NotImplementedError(
            f'getValue method undefined at this level: {type(self)}. Each expression must implement it.'
        )

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass

    @deprecated_parameters(
        obsolete_params={
            'numberOfDraws': 'number_of_draws',
            'prepareIds': 'prepare_ids',
        }
    )
    def get_value_c(
        self,
        database: Database | None = None,
        betas: dict[str, float] | None = None,
        number_of_draws: int = 1000,
        aggregation: bool = False,
        prepare_ids: bool = False,
    ) -> np.ndarray | float:
        """Evaluation of the expression, without the derivatives

        :param betas: values of the free parameters
        :type betas: list(float)

        :param database: database. If no database is provided, the
            expression must not contain any variable.
        :type database:  biogeme.database.Database

        :param number_of_draws: number of draws if needed by Monte-Carlo
            integration.
        :type number_of_draws: int

        :param aggregation: if a database is provided, and this
            parameter is True, the expression is applied on each entry
            of the database, and all values are aggregated, so that
            the sum is returned. If False, the list of all values is returned.
        :type aggregation: bool

        :param prepare_ids: if True, it means that the IDs of the
            expression must be constructed before the evaluation of
            the expression.
        :type prepare_ids: bool

        :return: if a database is provided, a list where each entry is
            the result of applying the expression on one entry of the
            database. It returns a float.

        :rtype: np.array or float

        :raise BiogemeError: if no database is given, and the number
            of returned values is different from one.

        """
        if self.requires_draws() and database is None:
            error_msg = (
                'An expression involving MonteCarlo integration '
                'must be associated with a database.'
            )
            raise BiogemeError(error_msg)

        the_function_output: (
            BiogemeDisaggregateFunctionOutput | BiogemeFunctionOutput
        ) = self.get_value_and_derivatives(
            betas=betas,
            database=database,
            number_of_draws=number_of_draws,
            gradient=False,
            hessian=False,
            bhhh=False,
            aggregation=aggregation,
            prepare_ids=prepare_ids,
        )

        if aggregation or database is None:
            return the_function_output.function
        return the_function_output.functions

    @deprecated(get_value_c)
    def getValue_c(
        self,
        database: Database | None = None,
        betas: dict[str, float] | None = None,
        number_of_draws: int = 1000,
        aggregation: bool = False,
        prepare_ids: bool = False,
    ):
        """Kept for backward compatibility"""
        pass

    @deprecated_parameters(
        obsolete_params={
            'numberOfDraws': 'number_of_draws',
            'prepareIds': 'prepare_ids',
        }
    )
    def get_value_and_derivatives(
        self,
        betas: dict[str, float] | None = None,
        database: Database | None = None,
        number_of_draws: int = 1000,
        gradient: bool = True,
        hessian: bool = True,
        bhhh: bool = True,
        aggregation: bool = True,
        prepare_ids: bool = False,
        named_results: bool = False,
    ) -> (
        BiogemeDisaggregateFunctionOutput
        | BiogemeFunctionOutput
        | NamedBiogemeDisaggregateFunctionOutput
        | NamedBiogemeFunctionOutput
    ):
        """Evaluation of the expression

        In Biogeme the complexity of some expressions requires a
        specific implementation, in C++. This function invokes the
        C++ code to evaluate the value of the expression for a
        series of entries in a database. Note that this function
        will generate draws if needed.

        :param betas: values of the free parameters
        :type betas: list(float)

        :param database: database. If no database is provided, the
            expression must not contain any variable.
        :type database:  biogeme.database.Database

        :param number_of_draws: number of draws if needed by Monte-Carlo
            integration.
        :type number_of_draws: int

        :param gradient: If True, the gradient is calculated.
        :type gradient: bool

        :param hessian: if True, the hessian is  calculated.
        :type hessian: bool

        :param bhhh: if True, the BHHH matrix is calculated.
        :type bhhh: bool

        :param aggregation: if a database is provided, and this
            parameter is True, the expression is applied on each entry
            of the database, and all values are aggregated, so that
            the sum is returned. If False, the list of all values is returned.
        :type aggregation: bool

        :param prepare_ids: if True, it means that the IDs of the
            expression must be constructed before the evaluation of
            the expression.
        :type prepare_ids: bool

        :param named_results: if True, the gradients, hessians, etc. are reported as dicts associating the names of
            the variables with their corresponding entry.

        :return: if a database is provided, a list where each entry is
            the result of applying the expression on one entry of the
            database. It returns a float, a vector, and a matrix,
            depending if derivatives are requested.

        :rtype: np.array or float, numpy.array, numpy.array

        :raise BiogemeError: if no database is given and the
            expressions involves variables.

        :raise BiogemeError: if gradient is False and hessian or BHHH is True.

        :raise BiogemeError: if derivatives are asked, and the expression
            is not simple.

        :raise BiogemeError: if the expression involves MonteCarlo integration,
           and no database is provided.
        """
        if prepare_ids:
            self.keep_id_manager = self.id_manager
            self.prepare(database, number_of_draws)
        elif self.id_manager is None:
            error_msg = 'Expression evaluated out of context. Set prepare_ids to True.'
            raise BiogemeError(error_msg)

        errors, warnings = self.audit(database)
        if warnings:
            logger.warning("\n".join(warnings))
        if errors:
            error_msg = "\n".join(errors)
            logger.warning(error_msg)
            raise BiogemeError(error_msg)

        if (hessian or bhhh) and not gradient:
            raise BiogemeError(
                "If the hessian or the BHHH matrix is calculated, "
                "so is the gradient. The provided parameters are inconsistent."
            )
        if database is None:
            variables = self.set_of_elementary_expression(
                TypeOfElementaryExpression.VARIABLE
            )
            if variables:
                raise BiogemeError(
                    f"No database is provided and the expression "
                    f"contains variables: {variables}"
                )

        self.numberOfDraws = number_of_draws

        if betas is not None:
            self.id_manager.free_betas_values = [
                (
                    betas[x]
                    if x in betas
                    else self.id_manager.free_betas.expressions[x].initValue
                )
                for x in self.id_manager.free_betas.names
            ]
            # List of values of the fixed Beta parameters (those not estimated)
            self.fixedBetaValues = [
                (
                    betas[x]
                    if x in betas
                    else self.id_manager.fixed_betas.expressions[x].initValue
                )
                for x in self.id_manager.fixed_betas.names
            ]
        results = calculate_function_and_derivatives(
            the_expression=self,
            database=database,
            calculate_gradient=gradient,
            calculate_hessian=hessian,
            calculate_bhhh=bhhh,
            aggregation=aggregation,
        )
        if named_results:
            if isinstance(results, BiogemeFunctionOutput) or isinstance(
                results, BiogemeFunctionOutputSmartOutputProxy
            ):
                results = NamedBiogemeFunctionOutput(
                    function_output=results, mapping=self.id_manager.free_betas.indices
                )
            elif isinstance(results, BiogemeDisaggregateFunctionOutput) or isinstance(
                results, BiogemeDisaggregateFunctionOutputSmartOutputProxy
            ):
                results = NamedBiogemeDisaggregateFunctionOutput(
                    function_output=results, mapping=self.id_manager.free_betas.indices
                )
            else:
                error_msg = f'Unknown type: {type(results)}'
                raise BiogemeError(error_msg)

        # Now, if we had to set the IDS, we reset them as they cannot
        # be used in another context.
        if prepare_ids:
            # We restore the previous Id manager
            self.set_id_manager(self.keep_id_manager)
        return results

    @deprecated(get_value_and_derivatives)
    def getValueAndDerivatives(
        self,
        betas: dict[str, float] | None = None,
        database: Database | None = None,
        number_of_draws: int = 1000,
        gradient: bool = True,
        hessian: bool = True,
        bhhh: bool = True,
        aggregation: bool = True,
        prepare_ids: bool = False,
    ) -> np.ndarray | FunctionOutput:
        """Kept for backward compatibility"""
        pass

    def requires_draws(self) -> bool:
        """Checks if the expression requires draws

        :return: True if it requires draws.
        :rtype: bool
        """
        return self.embed_expression("MonteCarlo")

    @deprecated(requires_draws)
    def requiresDraws(self) -> bool:
        """Kept for backward compatibility"""
        pass

    def get_beta_values(self) -> dict[str:float]:
        """Returns a dict with the initial values of Beta. Typically
            useful for simulation.

        :return: dict with the initial values of the Beta
        :rtype: dict(str: float)
        """

        betas = self.dict_of_elementary_expression(TypeOfElementaryExpression.FREE_BETA)
        return {b.name: b.initValue for b in betas.values()}

    def set_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> set[str]:
        """Extract a set with all elementary expressions of a specific type

        :param the_type: the type of expression

        :return: returns a set with the names of the elementary expressions


        """
        return set(self.dict_of_elementary_expression(the_type).keys())

    def dict_of_draw_types(self) -> dict[str:str]:
        """Extract a dict containing the types of draws involved in the expression"""
        the_draws = self.dict_of_elementary_expression(
            the_type=TypeOfElementaryExpression.DRAWS
        )
        return {name: expression.drawType for name, expression in the_draws.items()}

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str:Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression

        :return: returns a dict with the variables appearing in the
               expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        return dict(
            chain(
                *(
                    e.dict_of_elementary_expression(the_type).items()
                    for e in self.children
                )
            )
        )

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

    @deprecated(get_elementary_expression)
    def getElementaryExpression(self, name: str) -> Elementary | None:
        """Kept for backward compatibility"""
        pass

    def rename_elementary(
        self, names: Iterable[str], prefix: str | None = None, suffix: str | None = None
    ):
        """Rename elementary expressions by adding a prefix and/or a suffix

        :param names: names of expressions to rename
        :type names: list(str)

        :param prefix: if not None, the expression is renamed, with a
            prefix defined by this argument.
        :type prefix: str

        :param suffix: if not None, the expression is renamed, with a
            suffix defined by this argument.
        :type suffix: str
        """
        for e in self.get_children():
            e.rename_elementary(names, prefix=prefix, suffix=suffix)

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
        for e in self.get_children():
            e.fix_betas(beta_values, prefix=prefix, suffix=suffix)

    def get_class_name(self) -> str:
        """
        Obtain the name of the top class of the expression structure

        :return: the name of the class
        :rtype: string
        """
        n = type(self).__name__
        return n

    @deprecated(get_class_name)
    def getClassName(self) -> str:
        """Kept for backward compatibility"""
        pass

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of all the children expressions,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the number of children between ( )
            5. the ids of each children, preceeded by a comma.

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
        for e in self.get_children():
            list_of_signatures += e.get_signature()
        mysignature = f"<{self.get_class_name()}>"
        mysignature += f"{{{self.get_id()}}}"
        mysignature += f"({len(self.get_children())})"
        for e in self.get_children():
            mysignature += f",{e.get_id()}"
        list_of_signatures += [mysignature.encode()]
        return list_of_signatures

    @deprecated(get_signature)
    def getSignature(self) -> list[bytes]:
        """Kept for backward compatibility"""
        pass

    def embed_expression(self, t: str) -> bool:
        """Check if the expression contains an expression of type t.

        Typically, this would be used to check that a MonteCarlo
        expression contains a bioDraws expression.

        :return: True if the expression contains an expression of type t.
        :rtype: bool

        """
        if self.get_class_name() == t:
            return True
        for e in self.get_children():
            if e.embed_expression(t):
                return True
        return False

    @deprecated(embed_expression)
    def embedExpression(self, t: str) -> bool:
        """Kept for backward compatibility"""
        pass

    def count_panel_trajectory_expressions(self) -> int:
        """Count the number of times the PanelLikelihoodTrajectory
        is used in the formula. It should trigger an error if it
        is used more than once.

        :return: number of times the PanelLikelihoodTrajectory
            is used in the formula
        :rtype: int
        """
        nbr = 0
        for e in self.get_children():
            nbr += e.count_panel_trajectory_expressions()
        return nbr

    @deprecated(count_panel_trajectory_expressions)
    def countPanelTrajectoryExpressions(self) -> int:
        """Kept for backward compatibility"""

        pass

    def audit(self, database: Database | None = None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)
        """
        list_of_errors = []
        list_of_warnings = []

        for e in self.get_children():
            if not isinstance(e, Expression):
                the_error = f"Invalid expression: {e}"
                list_of_errors.append(the_error)
            err, war = e.audit(database)
            list_of_errors += err
            list_of_warnings += war

        return list_of_errors, list_of_warnings

    def change_init_values(self, betas: dict[str, float]):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        for e in self.get_children():
            e.change_init_values(betas)

    def dict_of_catalogs(
        self, ignore_synchronized: bool = False
    ) -> dict[str, Catalog]:
        """Returns a dict with all catalogs in the expression.

        :return: dict with all the catalogs
        """

        result = {}
        for e in self.children:
            a_dict = e.dict_of_catalogs(ignore_synchronized)
            for key, the_catalog in a_dict.items():
                result[key] = the_catalog
        return result

    def contains_catalog(self, name: str) -> bool:
        """Check if the expression contains a specific catalog

        :param name: name of the catalog to search.
        :type name: str

        :return: True if the given catalog is contained in the
            expression. False otherwise.
        :rtype: bool
        """
        all_catalogs = self.dict_of_catalogs()
        return name in all_catalogs

    def set_central_controller(
        self, the_central_controller: CentralController = None
    ) -> CentralController:
        """For multiple expressions, defines the central controller"""
        if the_central_controller is None:
            self.central_controller = CentralController(
                expression=self,
            )
        else:
            self.central_controller = the_central_controller

        for e in self.children:
            e.set_central_controller(self.central_controller)
        return self.central_controller

    def get_all_controllers(self) -> set[Controller]:
        """Provides all controllers  controlling the specifications of a multiple expression

        :return: a set of controllers
        :rtype: set(biogeme.controller.Controller)
        """
        if not self.children:
            return set()
        all_controllers = set()
        for e in self.children:
            all_controllers |= e.get_all_controllers()
        return all_controllers

    def number_of_multiple_expressions(self) -> int:
        """Reports the number of multiple expressions available through the iterator

        :return: number of  multiple expressions
        :rtype: int
        """
        if self.central_controller is None:
            self.set_central_controller()

        return self.central_controller.number_of_configurations()

    def set_of_configurations(self) -> set[str]:
        """Provides the set of all possible configurations"""
        if self.central_controller is None:
            self.set_central_controller()
        return self.central_controller.all_configurations

    def reset_expression_selection(self) -> None:
        """In each group of expressions, select the first one"""
        for e in self.children:
            e.reset_expression_selection()

    def configure_catalogs(self, configuration: Configuration):
        """Select the items in each catalog corresponding to the requested configuration

        :param configuration: catalog configuration
        :type configuration: biogeme.configuration.Configuration
        """
        if self.central_controller is None:
            self.set_central_controller()
        self.central_controller.set_configuration(configuration)

    def current_configuration(self) -> Configuration:
        """Obtain the current configuration of an expression

        :return: configuration
        :rtype: biogeme.configuration.Configuration
        """
        if self.central_controller is None:
            self.set_central_controller()

        return self.central_controller.get_configuration()

    def select_expression(self, controller_name: str, index: int):
        """Select a specific expression in a group

        :param controller_name: name of the controller
        :type controller_name: str

        :param index: index of the expression in the group
        :type index: int

        :raises BiogemeError: if index is out of range
        """
        if self.central_controller is None:
            self.set_central_controller()

        self.central_controller.set_controller(controller_name, index)

    def set_of_multiple_expressions(self) -> set[MultipleExpression]:
        """Set of the multiple expressions found in the current expression

        :return: a set of descriptions of the multiple expressions
        :rtype: set(MultipleExpressionDescription)
        """
        all_sets = [e.set_of_multiple_expressions() for e in self.get_children()]
        return set(chain.from_iterable(all_sets))

    def get_id(self) -> int:
        """Retrieve the id of the expression used in the signature

        :return: id of the object
        :rtype: int
        """
        return id(self)

    def get_children(self) -> list[Expression]:
        """Retrieve the list of children

        :return: list of children
        :rtype: list(Expression)
        """
        return self.children


ExpressionOrNumeric = Expression | float | int | bool
