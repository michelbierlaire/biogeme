""" Arithmetic expressions accepted by Biogeme: generic class

:author: Michel Bierlaire
:date: Sat Sep  9 15:25:07 2023
"""

from __future__ import annotations
import logging
from itertools import chain
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from biogeme.database import Database

import numpy as np
import biogeme.exceptions as excep
from biogeme_optimization.function import FunctionToMinimize, FunctionData
from biogeme.controller import CentralController
from .catalog_iterator import SelectedExpressionsIterator
from .idmanager import IdManager
from .numeric_tools import is_numeric
from .elementary_types import TypeOfElementaryExpression
from .calculator import calculate_function_and_derivatives

logger = logging.getLogger(__name__)


class Expression:
    """This is the general arithmetic expression in biogeme.
    It serves as a base class for concrete expressions.
    """

    def __init__(self):
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

    def getStatusIdManager(self) -> tuple[set[str], set[str]]:
        """Check the elementary expressions that are associated with
        an ID manager.

        :return: two sets of elementary expressions, those with and
            without an ID manager.
        """
        with_id = set()
        without_id = set()
        for e in self.get_children():
            yes, no = e.getStatusIdManager()
            with_id.update(yes)
            without_id.update(no)
        return with_id, without_id

    def prepare(self, database: "Database", numberOfDraws: int) -> None:
        """Prepare the expression to be evaluated

        :param database: Biogeme database

        :param numberOfDraws: number of draws for Monte-Carlo integration
        """
        # First, we reset the IDs, if any
        self.setIdManager(None)
        # Second, we calculate a new set of IDs.
        id_manager = IdManager([self], database, numberOfDraws)
        self.setIdManager(id_manager)

    def setIdManager(self, id_manager: IdManager) -> None:
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
            e.setIdManager(id_manager)

    def __repr__(self) -> str:
        """built-in function used to compute the 'official' string reputation
        of an object

        :return: description of the expression

        """
        return self.__str__()

    def __add__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added

        :return: self + other

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during addition to {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Plus

        return Plus(self, other)

    def __radd__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added
        :type other: biogeme.expressions.Expression

        :return: other + self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during addition to {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Plus

        return Plus(other, self)

    def __sub__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for substraction.

        :param other: expression to substract
        :type other: biogeme.expressions.Expression

        :return: self - other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during substraction to {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Minus

        return Minus(self, other)

    def __rsub__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for substraction.

        :param other: expression to be substracted
        :type other: biogeme.expressions.Expression

        :return: other - self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during substraction of {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Minus

        return Minus(other, self)

    def __mul__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for multiplication.

        :param other: expression to be multiplied
        :type other: biogeme.expressions.Expression

        :return: self * other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = (
                f"Invalid expression during multiplication " f"to {self}: [{other}]"
            )
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Times

        return Times(self, other)

    def __rmul__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for multiplication.

        :param other: expression to be multiplied
        :type other: biogeme.expressions.Expression

        :return: other * self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = (
                f"Invalid expression during multiplication " f"to {self}: [{other}]"
            )
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Times

        return Times(other, self)

    def __div__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: self / other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division of {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Divide

        return Divide(self, other)

    def __rdiv__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: other / self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division by {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Divide

        return Divide(other, self)

    def __truediv__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: self / other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division of {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
        from .binary_expressions import Divide

        return Divide(self, other)

    def __rtruediv__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: other / self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            error_msg = f"Invalid expression during division by {self}: [{other}]"
            raise excep.BiogemeError(error_msg)
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

    def __pow__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for power.

        :param other: expression for power
        :type other: biogeme.expressions.Expression

        :return: self ^ other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import Power

        return Power(self, other)

    def __rpow__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for power.

        :param other: expression for power
        :type other: biogeme.expressions.Expression

        :return: other ^ self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import Power

        return Power(other, self)

    def __and__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for logical and.

        :param other: expression for logical and
        :type other: biogeme.expressions.Expression

        :return: self and other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import And

        return And(self, other)

    def __rand__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for logical and.

        :param other: expression for logical and
        :type other: biogeme.expressions.Expression

        :return: other and self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import And

        return And(other, self)

    def __or__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for logical or.

        :param other: expression for logical or
        :type other: biogeme.expressions.Expression

        :return: self or other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import Or

        return Or(self, other)

    def __ror__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for logical or.

        :param other: expression for logical or
        :type other: biogeme.expressions.Expression

        :return: other or self
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .binary_expressions import Or

        return Or(other, self)

    def __eq__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for equality
        :type other: biogeme.expressions.Expression

        :return: self == other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .comparison_expressions import Equal

        return Equal(self, other)

    def __ne__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for difference
        :type other: biogeme.expressions.Expression

        :return: self != other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .comparison_expressions import NotEqual

        return NotEqual(self, other)

    def __le__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for less or equal
        :type other: biogeme.expressions.Expression

        :return: self <= other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .comparison_expressions import LessOrEqual

        return LessOrEqual(self, other)

    def __ge__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for greater or equal
        :type other: biogeme.expressions.Expression

        :return: self >= other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .comparison_expressions import GreaterOrEqual

        return GreaterOrEqual(self, other)

    def __lt__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for less than
        :type other: biogeme.expressions.Expression

        :return: self < other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .comparison_expressions import Less

        return Less(self, other)

    def __gt__(self, other: Expression) -> Expression:
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for greater than
        :type other: biogeme.expressions.Expression

        :return: self > other
        :rtype: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (is_numeric(other) or isinstance(other, Expression)):
            raise excep.BiogemeError(f"This is not a valid expression: {other}")
        from .comparison_expressions import Greater

        return Greater(self, other)

    def createFunction(
        self,
        database: Optional[Database] = None,
        numberOfDraws: Optional[int] = 1000,
        gradient: Optional[bool] = True,
        hessian: Optional[bool] = True,
        bhhh: Optional[bool] = False,
    ):
        """Create a function based on the expression. The function takes as
        argument an array for the free parameters, and return the
        value of the function, the gradient, the hessian and the BHHH. The
        calculation of the derivatives is optional.

        :param database: database. If no database is provided, the
            expression must not contain any variable.
        :type database:  biogeme.database.Database

        :param numberOfDraws: number of draws if needed by Monte-Carlo
            integration.
        :type numberOfDraws: int

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
            raise excep.BiogemeError(
                "If the hessian or BHHH is calculated, so is the gradient. "
                "The provided parameters are inconsistent."
            )

        with_id, without_id = self.getStatusIdManager()
        if len(without_id) > 0:
            if len(with_id) > 0:
                error_msg = (
                    f"IDs are defined for some expressions "
                    f"[{with_id}] but not for some [{without_id}]"
                )
                raise excep.BiogemeError(error_msg)
            self.setIdManager(IdManager([self], database, numberOfDraws))

        def my_function(x):
            if isinstance(x, (float, int, np.float64)):
                x = [float(x)]
            if len(x) != len(self.id_manager.free_betas_values):
                error_msg = (
                    f"Function is expecting an array of length "
                    f"{len(self.id_manager.free_betas_values)}, not {len(x)}"
                )
                raise excep.BiogemeError(error_msg)

            self.id_manager.free_betas_values = x
            f, g, h, b = self.getValueAndDerivatives(
                database=database,
                numberOfDraws=numberOfDraws,
                gradient=gradient,
                hessian=hessian,
                bhhh=bhhh,
                aggregation=True,
                prepareIds=False,
            )

            results = [f]
            if gradient:
                results.append(g)
                if hessian:
                    results.append(h)
                if bhhh:
                    results.append(b)
                return tuple(results)
            return f

        return my_function

    def create_objective_function(
        self,
        database=None,
        numberOfDraws=1000,
        gradient=True,
        hessian=True,
        bhhh=False,
    ):
        """Create a function based on the expression that complies
        with the interface of the biogeme_optimization model. The
        function takes as argument an array for the free parameters,
        and return the value of the function, the gradient, the
        hessian and the BHHH. The calculation of the derivatives is
        optional.

        :param database: database. If no database is provided, the
            expression must not contain any variable.
        :type database:  biogeme.database.Database

        :param numberOfDraws: number of draws if needed by Monte-Carlo
            integration.
        :type numberOfDraws: int

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
        expression_function = self.createFunction(
            database,
            numberOfDraws,
            gradient,
            hessian,
            bhhh,
        )

        class Function(FunctionToMinimize):
            """Class encapsulating the expression into a FunctioToMinimize"""

            def _f(self):
                f, g, h = expression_function(self.x)
                return f

            def _f_g(self):
                f, g, h = expression_function(self.x)
                return FunctionData(function=f, gradient=g, hessian=None)

            def _f_g_h(self):
                f, g, h = expression_function(self.x)
                return FunctionData(function=f, gradient=g, hessian=h)

            def dimension(self):
                return self.idmanager.number_of_free_betas

        return Function()

    def getValue(self) -> float:
        raise NotImplementedError(
            "getValue method undefined at this level. Each expression must implement it."
        )

    def getValue_c(
        self,
        database=None,
        betas=None,
        numberOfDraws=1000,
        aggregation=False,
        prepareIds=False,
    ):
        """Evaluation of the expression, without the derivatives

        :param betas: values of the free parameters
        :type betas: list(float)

        :param database: database. If no database is provided, the
            expression must not contain any variable.
        :type database:  biogeme.database.Database

        :param numberOfDraws: number of draws if needed by Monte-Carlo
            integration.
        :type numberOfDraws: int

        :param aggregation: if a database is provided, and this
            parameter is True, the expression is applied on each entry
            of the database, and all values are aggregated, so that
            the sum is returned. If False, the list of all values is returned.
        :type aggregation: bool

        :param prepareIds: if True, it means that the IDs of the
            expression must be constructed before the evaluation of
            the expression.
        :type prepareIds: bool

        :return: if a database is provided, a list where each entry is
            the result of applying the expression on one entry of the
            dsatabase. It returns a float.

        :rtype: np.array or float

        :raise BiogemeError: if no database is given, and the number
            of returned values is different from one.

        """
        if self.requiresDraws() and database is None:
            error_msg = (
                "An expression involving MonteCarlo integration "
                "must be associated with a database."
            )
            raise excep.BiogemeError(error_msg)

        f, _, _, _ = self.getValueAndDerivatives(
            betas=betas,
            database=database,
            numberOfDraws=numberOfDraws,
            gradient=False,
            hessian=False,
            bhhh=False,
            aggregation=aggregation,
            prepareIds=prepareIds,
        )
        if database is None:
            if len(f) != 1:
                error_msg = "Incorrect number of return values"
                raise excep.BiogemeError(error_msg)
            return f[0]
        return f

    def getValueAndDerivatives(
        self,
        betas=None,
        database=None,
        numberOfDraws=1000,
        gradient=True,
        hessian=True,
        bhhh=True,
        aggregation=True,
        prepareIds=False,
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

        :param numberOfDraws: number of draws if needed by Monte-Carlo
            integration.
        :type numberOfDraws: int

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

        :param prepareIds: if True, it means that the IDs of the
            expression must be constructed before the evaluation of
            the expression.
        :type prepareIds: bool

        :return: if a database is provided, a list where each entry is
            the result of applying the expression on one entry of the
            dsatabase. It returns a float, a vector, and a matrix,
            depedending if derivatives are requested.

        :rtype: np.array or float, numpy.array, numpy.array

        :raise BiogemeError: if no database is given and the
            expressions involves variables.

        :raise BiogemeError: if gradient is False and hessian or BHHH is True.

        :raise BiogemeError: if derivatives are asked, and the expression
            is not simple.

        :raise BiogemeError: if the expression involves MonteCarlo integration,
           and no database is provided.
        """
        if prepareIds:
            self.keep_id_manager = self.id_manager
            self.prepare(database, numberOfDraws)
        elif self.id_manager is None:
            error_msg = "Expression evaluated out of context. Set prepareIds to True."
            raise excep.BiogemeError(error_msg)

        errors, warnings = self.audit(database)
        if warnings:
            logger.warning("\n".join(warnings))
        if errors:
            error_msg = "\n".join(errors)
            logger.warning(error_msg)
            raise excep.BiogemeError(error_msg)

        if (hessian or bhhh) and not gradient:
            raise excep.BiogemeError(
                "If the hessian or the BHHH matrix is calculated, "
                "so is the gradient. The provided parameters are inconsistent."
            )
        if database is None:
            variables = self.set_of_elementary_expression(
                TypeOfElementaryExpression.VARIABLE
            )
            if variables:
                raise excep.BiogemeError(
                    f"No database is provided and the expression "
                    f"contains variables: {variables}"
                )

        self.numberOfDraws = numberOfDraws

        if betas is not None:
            self.id_manager.free_betas_values = [
                betas[x]
                if x in betas
                else self.id_manager.free_betas.expressions[x].initValue
                for x in self.id_manager.free_betas.names
            ]
            # List of values of the fixed beta parameters (those not estimated)
            self.fixedBetaValues = [
                betas[x]
                if x in betas
                else self.id_manager.fixed_betas.expressions[x].initValue
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

        # Now, if we had to set the IDS, we reset them as they cannot
        # be used in another context.
        if prepareIds:
            # We restore the previous Id manager
            self.setIdManager(self.keep_id_manager)
        return results

    def requiresDraws(self):
        """Checks if the expression requires draws

        :return: True if it requires draws.
        :rtype: bool
        """
        return self.embedExpression("MonteCarlo")

    def get_beta_values(self):
        """Returns a dict with the initial values of beta. Typically
            useful for simulation.

        :return: dict with the initial values of the beta
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

    def dict_of_elementary_expression(self, the_type):
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

    def getElementaryExpression(self, name):
        """Return: an elementary expression from its name if it appears in the
        expression.

        :param name: name of the elementary expression.
        :type name: string

        :return: the expression if it exists. None otherwise.
        :rtype: biogeme.expressions.Expression
        """
        for e in self.get_children():
            if e.getElementaryExpression(name) is not None:
                return e.getElementaryExpression(name)
        return None

    def setRow(self, row):
        """Obsolete function.
        This function identifies the row of the database from which the
        values of the variables must be obtained.

        :param row: row from the database
        :type row: pandas.core.series.Serie

        :raise BiogemeError: if the function is called, because it is obsolete.
        """
        raise excep.BiogemeError("The function setRow is now obsolete.")

    def rename_elementary(self, names, prefix=None, suffix=None):
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

    def fix_betas(self, beta_values, prefix=None, suffix=None):
        """Fix all the values of the beta parameters appearing in the
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

    def getClassName(self):
        """
        Obtain the name of the top class of the expression structure

        :return: the name of the class
        :rtype: string
        """
        n = type(self).__name__
        return n

    def getSignature(self):
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
        listOfSignatures = []
        for e in self.get_children():
            listOfSignatures += e.getSignature()
        mysignature = f"<{self.getClassName()}>"
        mysignature += f"{{{self.get_id()}}}"
        mysignature += f"({len(self.get_children())})"
        for e in self.get_children():
            mysignature += f",{e.get_id()}"
        listOfSignatures += [mysignature.encode()]
        return listOfSignatures

    def embedExpression(self, t):
        """Check if the expression contains an expression of type t.

        Typically, this would be used to check that a MonteCarlo
        expression contains a bioDraws expression.

        :return: True if the expression contains an expression of type t.
        :rtype: bool

        """
        if self.getClassName() == t:
            return True
        for e in self.get_children():
            if e.embedExpression(t):
                return True
        return False

    def countPanelTrajectoryExpressions(self):
        """Count the number of times the PanelLikelihoodTrajectory
        is used in the formula. It should trigger an error if it
        is used more than once.

        :return: number of times the PanelLikelihoodTrajectory
            is used in the formula
        :rtype: int
        """
        nbr = 0
        for e in self.get_children():
            nbr += e.countPanelTrajectoryExpressions()
        return nbr

    def audit(self, database=None):
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

    def change_init_values(self, betas):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        for e in self.get_children():
            e.change_init_values(betas)

    def set_estimated_values(self, betas: dict[str, float]):
        """Set the estimated values of beta

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        """
        for e in self.get_children():
            e.set_estimated_values(betas)

    def dict_of_catalogs(self, ignore_synchronized=False):
        """Returns a dict with all catalogs in the expression.

        :return: dict with all the catalogs
        """

        result = {}
        for e in self.children:
            a_dict = e.dict_of_catalogs(ignore_synchronized)
            for key, the_catalog in a_dict.items():
                result[key] = the_catalog
        return result

    def contains_catalog(self, name):
        """Check if the expression contains a specific catalog

        :param name: name of the catalog to search.
        :type name: str

        :return: True if the given catalog is contained in the
            expression. False otherwise.
        :rtype: bool
        """
        all_catalogs = self.dict_of_catalogs()
        return name in all_catalogs

    def set_central_controller(self, the_central_controller=None):
        if the_central_controller is None:
            self.central_controller = CentralController(
                expression=self,
            )
        else:
            self.central_controller = the_central_controller

        for e in self.children:
            e.set_central_controller(self.central_controller)
        return self.central_controller

    def get_all_controllers(self):
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

    def number_of_multiple_expressions(self):
        """Reports the number of multiple expressions available through the iterator

        :return: number of  multiple expressions
        :rtype: int
        """
        if self.central_controller is None:
            self.set_central_controller()

        return self.central_controller.number_of_configurations()

    def set_of_configurations(self):
        """Provides the set of all possible configurations"""
        if self.central_controller is None:
            self.set_central_controller()
        return self.central_controller.all_configurations

    def reset_expression_selection(self):
        """In each group of expressions, select the first one"""
        for e in self.children:
            e.reset_expression_selection()

    def configure_catalogs(self, configuration):
        """Select the items in each catalog corresponding to the requested configuration

        :param configuration: catalog configuration
        :type configuration: biogeme.configuration.Configuration
        """
        if self.central_controller is None:
            self.set_central_controller()
        self.central_controller.set_configuration(configuration)

    def current_configuration(self):
        """Obtain the current configuration of an expression

        :return: configuration
        :rtype: biogeme.configuration.Configuration
        """
        if self.central_controller is None:
            self.set_central_controller()

        return self.central_controller.get_configuration()

    def select_expression(self, controller_name, index):
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

    def set_of_multiple_expressions(self):
        """Set of the multiple expressions found in the current expression

        :return: a set of descriptions of the multiple expressions
        :rtype: set(MultipleExpressionDescription)
        """
        all_sets = [e.set_of_multiple_expressions() for e in self.get_children()]
        return set(chain.from_iterable(all_sets))

    def get_id(self):
        """Retrieve the id of the expression used in the signature

        :return: id of the object
        :rtype: int
        """
        return id(self)

    def get_children(self):
        """Retrieve the list of children

        :return: list of children
        :rtype: list(Expression)
        """
        return self.children


def number_to_expression(x: Union[float, int, Expression]) -> Expression:
    """If x is a number, the corresponding expression is
    returned. Otherwise, if x is not an Expression, an exception is
    raised.

    :param x: value to convert to an Expression

    :return: if x is an expression, it is simply retirned. If not, it
        is converted into an expression.

    :raise BiogemeError: if x is neither an expression or a number.
    """
    if is_numeric(x):
        return Numeric(x)

    if isinstance(x, Expression):
        return x

    raise BiogemeError(
        f'Object of type {type(x)} can not be treated. Expect an Expression or a number.'
    )
