""" Defines the various arithmetic expressions accepted by Biogeme.

:author: Michel Bierlaire

:date: Tue Mar 26 16:47:49 2019

"""

# Too constraining
# pylint: disable=invalid-name

# Too constraining
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
# pylint: disable=too-many-instance-attributes, too-many-lines

from itertools import chain
import numpy as np
import biogeme.exceptions as excep
import biogeme.messaging as msg
import biogeme.cythonbiogeme as ee
from biogeme.idmanager import IdManager

logger = msg.bioMessage()


def isNumeric(obj):
    """Identifies if an object is numeric, that is int, float or bool.

    :param obj: any object
    :type obj: object

    :return: True if the object is int, float or bool.
    :rtype: bool
    """
    return isinstance(obj, (int, float, bool))


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

        self.cpp = ee.pyEvaluateOneExpression()
        """ Interface to the C++ implementation
        """

        self.missingData = 99999
        """ Value interpreted as missing data
        """

    def check_panel_trajectory(self):
        """Set of variables defined outside of 'PanelLikelihoodTrajectory'

        :return: List of names of variables
        :rtype: set(str)
        """
        check_children = set(
            chain.from_iterable(
                [e.check_panel_trajectory() for e in self.children]
            )
        )
        return check_children

    def check_draws(self):
        """Set of draws defined outside of 'MonteCarlo'

        :return: List of names of variables
        :rtype: set(str)
        """
        check_children = set(
            chain.from_iterable([e.check_draws() for e in self.children])
        )
        return check_children

    def check_rv(self):
        """Set of random variables defined outside of 'Integrate'

        :return: List of names of variables
        :rtype: set(str)
        """
        check_children = set(
            chain.from_iterable([e.check_rv() for e in self.children])
        )
        return check_children

    def getStatusIdManager(self):
        """Check the elementary expressions that are associated with
        an ID manager.

        :return: two sets of elementary expressions, those with and
            without an ID manager.
        :rtype: tuple(set(str), set(str))
        """
        with_id = set()
        without_id = set()
        for e in self.children:
            yes, no = e.getStatusIdManager()
            with_id.update(yes)
            without_id.update(no)
        return with_id, without_id

    def prepare(self, database, numberOfDraws=1000):
        """Prepare the expression to be evaluated

        :param database: Biogeme database
        :type database: biogeme.database.Database

        :param numberOfDraws: number of draws for Monte-Carlo integration
        :type numberOfDraws: int
        """
        # First, we reset the IDs, if any
        self.setIdManager(None)
        # Second, we calculate a new set of IDs.
        id_manager = IdManager([self], database, numberOfDraws)
        self.setIdManager(id_manager)

    def setIdManager(self, id_manager):
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may nee to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """
        self.id_manager = id_manager
        for e in self.children:
            e.setIdManager(id_manager)

    def __repr__(self):
        """built-in function used to compute the 'official' string reputation
        of an object

        :return: description of the expression
        :rtype: string

        """
        return self.__str__()

    def __add__(self, other):
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added
        :type other: biogeme.expressions.Expression

        :return: self + other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during addition to {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Plus(self, other)

    def __radd__(self, other):
        """
        Operator overloading. Generate an expression for addition.

        :param other: expression to be added
        :type other: biogeme.expressions.Expression

        :return: other + self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during addition to {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Plus(other, self)

    def __sub__(self, other):
        """
        Operator overloading. Generate an expression for substraction.

        :param other: expression to substract
        :type other: biogeme.expressions.Expression

        :return: self - other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during substraction to {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Minus(self, other)

    def __rsub__(self, other):
        """
        Operator overloading. Generate an expression for substraction.

        :param other: expression to be substracted
        :type other: biogeme.expressions.Expression

        :return: other - self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during substraction of {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Minus(other, self)

    def __mul__(self, other):
        """
        Operator overloading. Generate an expression for multiplication.

        :param other: expression to be multiplied
        :type other: biogeme.expressions.Expression

        :return: self * other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during multiplication '
                f'to {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Times(self, other)

    def __rmul__(self, other):
        """
        Operator overloading. Generate an expression for multiplication.

        :param other: expression to be multiplied
        :type other: biogeme.expressions.Expression

        :return: other * self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during multiplication '
                f'to {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Times(other, self)

    def __div__(self, other):
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: self / other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during division of {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Divide(self, other)

    def __rdiv__(self, other):
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: other / self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during division by {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Divide(other, self)

    def __truediv__(self, other):
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: self / other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during division of {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Divide(self, other)

    def __rtruediv__(self, other):
        """
        Operator overloading. Generate an expression for division.

        :param other: expression for division
        :type other: biogeme.expressions.Expression

        :return: other / self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            error_msg = (
                f'Invalid expression during division by {self}: [{other}]'
            )
            raise excep.biogemeError(error_msg)
        return Divide(other, self)

    def __neg__(self):
        """
        Operator overloading. Generate an expression for unary minus.

        :return: -self
        :rtype: biogeme.expressions.Expression
        """
        return UnaryMinus(self)

    def __pow__(self, other):
        """
        Operator overloading. Generate an expression for power.

        :param other: expression for power
        :type other: biogeme.expressions.Expression

        :return: self ^ other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return Power(self, other)

    def __rpow__(self, other):
        """
        Operator overloading. Generate an expression for power.

        :param other: expression for power
        :type other: biogeme.expressions.Expression

        :return: other ^ self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return Power(other, self)

    def __and__(self, other):
        """
        Operator overloading. Generate an expression for logical and.

        :param other: expression for logical and
        :type other: biogeme.expressions.Expression

        :return: self and other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return And(self, other)

    def __rand__(self, other):
        """
        Operator overloading. Generate an expression for logical and.

        :param other: expression for logical and
        :type other: biogeme.expressions.Expression

        :return: other and self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return And(other, self)

    def __or__(self, other):
        """
        Operator overloading. Generate an expression for logical or.

        :param other: expression for logical or
        :type other: biogeme.expressions.Expression

        :return: self or other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return Or(self, other)

    def __ror__(self, other):
        """
        Operator overloading. Generate an expression for logical or.

        :param other: expression for logical or
        :type other: biogeme.expressions.Expression

        :return: other or self
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return Or(other, self)

    def __eq__(self, other):
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for equality
        :type other: biogeme.expressions.Expression

        :return: self == other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return Equal(self, other)

    def __ne__(self, other):
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for difference
        :type other: biogeme.expressions.Expression

        :return: self != other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return NotEqual(self, other)

    def __le__(self, other):
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for less or equal
        :type other: biogeme.expressions.Expression

        :return: self <= other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return LessOrEqual(self, other)

    def __ge__(self, other):
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for greater or equal
        :type other: biogeme.expressions.Expression

        :return: self >= other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return GreaterOrEqual(self, other)

    def __lt__(self, other):
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for less than
        :type other: biogeme.expressions.Expression

        :return: self < other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return Less(self, other)

    def __gt__(self, other):
        """
        Operator overloading. Generate an expression for comparison.

        :param other: expression for greater than
        :type other: biogeme.expressions.Expression

        :return: self > other
        :rtype: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        if not (isNumeric(other) or isinstance(other, Expression)):
            raise excep.biogemeError(
                f'This is not a valid expression: {other}'
            )
        return Greater(self, other)

    def createFunction(
        self,
        database=None,
        numberOfDraws=1000,
        gradient=True,
        hessian=True,
        bhhh=False,
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

        :raise biogemeError: if gradient is False and hessian or BHHH is True.

        """
        if (hessian or bhhh) and not gradient:
            raise excep.biogemeError(
                'If the hessian or BHHH is calculated, so is the gradient. '
                'The provided parameters are inconsistent.'
            )

        with_id, without_id = self.getStatusIdManager()
        if len(without_id) > 0:
            if len(with_id) > 0:
                error_msg = (
                    f'IDs are defined for some expressions '
                    f'[{with_id}] but not for some [{without_id}]'
                )
                raise excep.biogemeError(error_msg)
            self.setIdManager(IdManager([self], database, numberOfDraws))

        def my_function(x):
            if isinstance(x, (float, int, np.float64)):
                x = [float(x)]
            if len(x) != len(self.id_manager.free_betas_values):
                error_msg = (
                    f'Function is expecting an array of length '
                    f'{len(self.id_manager.free_betas_values)}, not {len(x)}'
                )
                excep.biogemeError(error_msg)

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

        :raise biogemeError: if no database is given, and the number
            of returned values is different from one.

        """
        if self.requiresDraws() and database is None:
            error_msg = (
                'An expression involving MonteCarlo integration '
                'must be associated with a database.'
            )
            raise excep.biogemeError(error_msg)

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
                error_msg = 'Incorrect number of return values'
                raise excep.biogemeError(error_msg)
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

        :raise biogemeError: if no database is given and the
            expressions involves variables.

        :raise biogemeError: if gradient is False and hessian or BHHH is True.

        :raise biogemeError: if derivatives are asked, and the expression
            is not simple.

        :raise biogemeError: if the expression involves MonteCarlo integration,
           and no database is provided.
        """
        if prepareIds:
            self.keep_id_manager = self.id_manager
            self.prepare(database, numberOfDraws)
        elif self.id_manager is None:
            error_msg = (
                'Expression evaluated out of context. Set prepareIds to True.'
            )
            raise excep.biogemeError(error_msg)

        errors, warnings = self.audit(database)
        if warnings:
            logger.warning('\n'.join(warnings))
        if errors:
            error_msg = '\n'.join(errors)
            logger.warning(error_msg)
            raise excep.biogemeError(error_msg)

        if (hessian or bhhh) and not gradient:
            raise excep.biogemeError(
                'If the hessian or the BHHH matrix is calculated, '
                'so is the gradient. The provided parameters are inconsistent.'
            )
        if database is None:
            variables = self.setOfVariables()
            if variables:
                raise excep.biogemeError(
                    f'No database is provided and the expression '
                    f'contains variables: {variables}'
                )

        self.numberOfDraws = numberOfDraws

        if database is not None:
            self.cpp.setData(database.data)
            if self.embedExpression('PanelLikelihoodTrajectory'):
                if database.isPanel():
                    database.buildPanelMap()
                    self.cpp.setDataMap(database.individualMap)
                else:
                    error_msg = (
                        'The expression involves '
                        '"PanelLikelihoodTrajectory" '
                        'that requires panel data'
                    )
                    raise excep.biogemeError(error_msg)

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

        self.cpp.setExpression(self.getSignature())
        self.cpp.setFreeBetas(self.id_manager.free_betas_values)
        self.cpp.setFixedBetas(self.id_manager.fixed_betas_values)
        self.cpp.setMissingData(self.missingData)

        if self.requiresDraws():
            if database is None:
                error_msg = (
                    'An expression involving MonteCarlo integration '
                    'must be associated with a database.'
                )
                raise excep.biogemeError(error_msg)
            self.cpp.setDraws(database.theDraws)

        self.cpp.calculate(
            gradient=gradient,
            hessian=hessian,
            bhhh=bhhh,
            aggregation=aggregation,
        )

        f, g, h, b = self.cpp.getResults()

        gres = g if gradient else None
        hres = h if hessian else None
        bhhhres = b if bhhh else None

        if aggregation:
            results = (
                f[0],
                None if gres is None else g[0],
                None if hres is None else h[0],
                None if bhhhres is None else b[0],
            )
        else:
            results = (f, gres, hres, bhhhres)

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
        return self.embedExpression('MonteCarlo')

    def setOfBetas(self, free=True, fixed=False):
        """
        Extract the set of parameters from the expression.

        :param free: if True, the free parameters are included. Default: True.
        :type free: bool
        :param fixed: if True, the fixed parameters are included.
            Default: False.
        :type fixed: bool
        :return: returns a set with the beta parameters appearing in the
            expression.
        :rtype: set(biogeme.expressions.Expression)
        """
        s = set()
        for e in self.children:
            s = s.union(e.setOfBetas(free, fixed))
        return s

    def setOfVariables(self):
        """
        Extract the set of variables used in the expression.

        :return: returns a set with the variables appearing in the expression.
        :rtype: set(biogeme.expressions.Expression)
        """
        return set(self.dictOfVariables().keys())

    def dictOfBetas(self, free=True, fixed=False):

        """
        Extract the set of parameters from the expression.

        :param free: if True, the free parameters are included. Default: True.
        :type free: bool
        :param fixed: if True, the fixed parameters are included.
             Default: False.
        :type fixed: bool

        :return: a dict with the beta parameters appearing in the expression,
                 the keys being the names of the parameters.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        s = {}
        for e in self.children:
            d = e.dictOfBetas(free, fixed)
            s = dict(s, **d)
        return s

    def dictOfVariables(self):
        """Recursively extract the variables appearing in the expression, and
        store them in a dictionary.

        :return: returns a dict with the variables appearing in the
                 expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        s = {}
        for e in self.children:
            d = e.dictOfVariables()
            s = dict(s, **d)
        return s

    def dictOfRandomVariables(self):
        """Recursively extract the random variables appearing in the
        expression, and store them in a dictionary.

        :return: returns a dict with the random variables appearing in
                 the expression the keys being their names.

        :rtype: dict(string:biogeme.expressions.Expression)
        """
        s = {}
        for e in self.children:
            d = e.dictOfRandomVariables()
            s = dict(s, **d)
        return s

    def getElementaryExpression(self, name):
        """Return: an elementary expression from its name if it appears in the
        expression.

        :param name: name of the elementary expression.
        :type name: string

        :return: the expression if it exists. None otherwise.
        :rtype: biogeme.expressions.Expression
        """
        for e in self.children:
            if e.getElementaryExpression(name) is not None:
                return e.getElementaryExpression(name)
        return None

    def setRow(self, row):
        """Obsolete function.
        This function identifies the row of the database from which the
        values of the variables must be obtained.

        :param row: row from the database
        :type row: pandas.core.series.Serie

        :raise biogemeError: if the function is called, because it is obsolete.
        """
        raise excep.biogemeError("The function setRow is now obsolete.")

    def dictOfDraws(self):
        """Recursively extract the random variables
        (draws for Monte-Carlo)
        appearing in the expression, and store them in a dictionary.

        :return: dict where the keys are the random variables and the elements
             the type of draws
        :rtype: dict(string:string)
        """
        draws = {}
        for e in self.children:
            d = e.dictOfDraws()
            if d:
                draws = dict(draws, **d)
        return draws

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
        for e in self.children:
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
        for e in self.children:
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
        for e in self.children:
            listOfSignatures += e.getSignature()
        mysignature = f'<{self.getClassName()}>'
        mysignature += f'{{{id(self)}}}'
        mysignature += f'({len(self.children)})'
        for e in self.children:
            mysignature += f',{id(e)}'
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
        for e in self.children:
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
        for e in self.children:
            nbr += e.countPanelTrajectoryExpressions()
        return nbr

    def audit(self, database=None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)
        """
        listOfErrors = []
        listOfWarnings = []

        for e in self.children:
            if not isinstance(e, Expression):
                theError = f'Invalid expression: {e}'
                listOfErrors.append(theError)
            err, war = e.audit(database)
            listOfErrors += err
            listOfWarnings += war
        return listOfErrors, listOfWarnings

    def changeInitValues(self, betas):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        for e in self.children:
            e.changeInitValues(betas)


class BinaryOperator(Expression):
    """
    Base class for arithmetic expressions that are binary operators.
    This expression is the result of the combination of two expressions,
    typically addition, substraction, multiplication or division.
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        Expression.__init__(self)
        if isNumeric(left):
            self.left = Numeric(left)  #: left child
        else:
            if not isinstance(left, Expression):
                raise excep.biogemeError(
                    f'This is not a valid expression: {left}'
                )
            self.left = left
        if isNumeric(right):
            self.right = Numeric(right)  #: right child
        else:
            if not isinstance(right, Expression):
                raise excep.biogemeError(
                    f'This is not a valid expression: {right}'
                )
            self.right = right
        self.children.append(self.left)
        self.children.append(self.right)


class Plus(BinaryOperator):
    """
    Addition expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} + {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() + self.right.getValue()


class Minus(BinaryOperator):
    """
    Substraction expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} - {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() - self.right.getValue()


class Times(BinaryOperator):
    """
    Multiplication expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} * {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() * self.right.getValue()


class Divide(BinaryOperator):
    """
    Division expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} / {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() / self.right.getValue()


class Power(BinaryOperator):
    """
    Power expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} ** {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() ** self.right.getValue()


class bioMin(BinaryOperator):
    """
    Minimum of two expressions
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'bioMin({self.left}, {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() <= self.right.getValue():
            return self.left.getValue()

        return self.right.getValue()


class bioMax(BinaryOperator):
    """
    Maximum of two expressions
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'bioMax({self.left}, {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() >= self.right.getValue():
            return self.left.getValue()

        return self.right.getValue()


class And(BinaryOperator):
    """
    Logical and
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} and {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() == 0.0:
            return 0.0
        if self.right.getValue() == 0.0:
            return 0.0
        return 1.0


class Or(BinaryOperator):
    """
    Logical or
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} or {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() != 0.0:
            return 1.0
        if self.right.getValue() != 0.0:
            return 1.0
        return 0.0


class ComparisonOperator(BinaryOperator):
    """Base class for comparison expressions."""

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def audit(self, database=None):
        """Performs various checks on the expression."""
        listOfErrors = []
        listOfWarnings = []
        if isinstance(self.left, ComparisonOperator) or isinstance(
            self.right, ComparisonOperator
        ):
            print(f'Current expression: {self}')
            print(
                f'Left expression: {self.left} '
                f'[{isinstance(self.left, ComparisonOperator)}]'
            )
            print(
                f'Right expression: {self.right} '
                f'[{isinstance(self.right, ComparisonOperator)}]'
            )
            print(f'Type left expression: {type(self.left)}')
            the_warning = (
                f'Chaining two comparisons expressions is not recommended'
                f' as it may be ambiguous. '
                f'Keep in mind that, for Biogeme, the '
                f'expression (a <= x <= b) is not equivalent to (a <= x) '
                f'and (x <= b) [{self}]'
            )
            listOfWarnings.append(the_warning)
        return listOfErrors, listOfWarnings


class Equal(ComparisonOperator):
    """
    Logical equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} == {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() == self.right.getValue() else 0
        return r


class NotEqual(ComparisonOperator):
    """
    Logical not equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} != {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() != self.right.getValue() else 0
        return r


class LessOrEqual(ComparisonOperator):
    """
    Logical less or equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} <= {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() <= self.right.getValue() else 0
        return r


class GreaterOrEqual(ComparisonOperator):
    """
    Logical greater or equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} >= {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() >= self.right.getValue() else 0
        return r


class Less(ComparisonOperator):
    """
    Logical less
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} < {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() < self.right.getValue() else 0
        return r


class Greater(ComparisonOperator):
    """
    Logical greater
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} > {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() > self.right.getValue() else 0
        return r


class UnaryOperator(Expression):
    """
    Base class for arithmetic expressions that are unary operators.

    Such an expression is the result of the modification of another
    expressions, typically changing its sign.
    """

    def __init__(self, child):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        Expression.__init__(self)
        if isNumeric(child):
            self.child = Numeric(child)  #: child
        else:
            if not isinstance(child, Expression):
                raise excep.biogemeError(
                    f'This is not a valid expression: {child}'
                )
            self.child = child
        self.children.append(self.child)


class UnaryMinus(UnaryOperator):
    """
    Unary minus expression
    """

    def __init__(self, child):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self):
        return f'(-{self.child})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return -self.child.getValue()


class MonteCarlo(UnaryOperator):
    """
    Monte Carlo integration
    """

    def __init__(self, child):
        """Constructor

        :param child: arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self):
        return f'MonteCarlo({self.child})'

    def check_draws(self):
        """List of draws defined outside of 'MonteCarlo'

        :return: List of names of variables
        :rtype: list(str)
        """
        return set()

    def audit(self, database=None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)

        """
        listOfErrors, listOfWarnings = self.child.audit(database)
        if database is None:
            if self.child.embedExpression('PanelLikelihoodTrajectory'):
                theWarning = (
                    'The formula contains a PanelLikelihoodTrajectory '
                    'expression, and no database is given'
                )
                listOfWarnings.append(theWarning)
        else:
            if database.isPanel() and not self.child.embedExpression(
                'PanelLikelihoodTrajectory'
            ):
                theError = (
                    f'As the database is panel, the argument '
                    f'of MonteCarlo must contain a'
                    f' PanelLikelihoodTrajectory: {self}'
                )
                listOfErrors.append(theError)

        if not self.child.embedExpression('bioDraws'):
            theError = (
                f'The argument of MonteCarlo must contain a'
                f' bioDraws: {self}'
            )
            listOfErrors.append(theError)
        if self.child.embedExpression('MonteCarlo'):
            theError = (
                f'It is not possible to include a MonteCarlo '
                f'statement in another one: {self}'
            )
            listOfErrors.append(theError)
        return listOfErrors, listOfWarnings


class bioNormalCdf(UnaryOperator):
    """
    Cumulative Distribution Function of a normal random variable
    """

    def __init__(self, child):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self):
        return f'bioNormalCdf({self.child})'


class PanelLikelihoodTrajectory(UnaryOperator):
    """
    Likelihood of a sequences of observations for the same individual
    """

    def __init__(self, child):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self):
        return f'PanelLikelihoodTrajectory({self.child})'

    def check_panel_trajectory(self):
        """List of variables defined outside of 'PanelLikelihoodTrajectory'

        :return: List of names of variables
        :rtype: list(str)
        """
        return set()

    def countPanelTrajectoryExpressions(self):
        """Count the number of times the PanelLikelihoodTrajectory
        is used in the formula.
        """
        return 1 + self.child.countPanelTrajectoryExpressions()

    def audit(self, database=None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)

        """
        listOfErrors, listOfWarnings = self.child.audit(database)
        if not database.isPanel():
            theError = (
                f'Expression PanelLikelihoodTrajectory can '
                f'only be used with panel data. Use the statement '
                f'database.panel("IndividualId") to declare the '
                f'panel structure of the data: {self}'
            )
            listOfErrors.append(theError)
        return listOfErrors, listOfWarnings


class exp(UnaryOperator):
    """
    exponential expression
    """

    def __init__(self, child):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self):
        return f'exp({self.child})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.exp(self.child.getValue())


class log(UnaryOperator):
    """
    logarithm expression
    """

    def __init__(self, child):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)

    def __str__(self):
        return f'log({self.child})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.log(self.child.getValue())


class Derive(UnaryOperator):
    """
    Derivative with respect to an elementary expression
    """

    def __init__(self, child, name):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        UnaryOperator.__init__(self, child)
        # Name of the elementary expression by which the derivative is taken
        self.elementaryName = name

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of the child expression,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the id of the child, preceeded by a comma.

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
        elementaryIndex = self.id_manager.elementary_expressions.indices[
            self.elementaryName
        ]
        listOfSignatures = []
        listOfSignatures += self.child.getSignature()
        mysignature = f'<{self.getClassName()}>'
        mysignature += f'{{{id(self)}}}'
        mysignature += f',{id(self.child)}'
        mysignature += f',{elementaryIndex}'
        listOfSignatures += [mysignature.encode()]
        return listOfSignatures

    def __str__(self):
        return 'Derive({self.child}, "{self.elementName}")'


class Integrate(UnaryOperator):
    """
    Numerical integration
    """

    def __init__(self, child, name):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        :param name: name of the random variable for the integration.
        :type name: string
        """
        UnaryOperator.__init__(self, child)
        self.randomVariableName = name

    def check_rv(self):
        """List of random variables defined outside of 'Integrate'

        :return: List of names of variables
        :rtype: list(str)
        """
        return set()

    def audit(self, database=None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)

        """
        listOfErrors, listOfWarnings = self.child.audit(database)
        if not self.child.embedExpression('RandomVariable'):
            theError = (
                f'The argument of Integrate must contain a '
                f'RandomVariable: {self}'
            )
            listOfErrors.append(theError)
        return listOfErrors, listOfWarnings

    def getSignature(self):
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
        randomVariableIndex = self.id_manager.random_variables.indices[
            self.randomVariableName
        ]
        listOfSignatures = []
        listOfSignatures += self.child.getSignature()
        mysignature = f'<{self.getClassName()}>'
        mysignature += f'{{{id(self)}}}'
        mysignature += f',{id(self.child)}'
        mysignature += f',{randomVariableIndex}'
        listOfSignatures += [mysignature.encode()]
        return listOfSignatures

    def __str__(self):
        return f'Integrate({self.child}, "{self.randomVariableName}")'


class Elementary(Expression):
    """Elementary expression.

    It is typically defined by a name appearing in an expression. It
    can be a variable (from the database), or a parameter (fixed or to
    be estimated using maximum likelihood), a random variable for
    numrerical integration, or Monte-Carlo integration.

    """

    def __init__(self, name):
        """Constructor

        :param name: name of the elementary experession.
        :type name: string

        """
        Expression.__init__(self)
        self.name = name  #: name of the elementary expressiom

        self.elementaryIndex = None
        """The id should be unique for all elementary expressions
        appearing in a given set of formulas.
        """

    def __str__(self):
        """string method

        :return: name of the expression
        :rtype: str
        """
        return f'{self.name}'

    def getStatusIdManager(self):
        """Check the elementary expressions that are associated with
        an ID manager.

        :return: two lists of elementary expressions, those with and
            without an ID manager.
        :rtype: tuple(list(str), list(str))
        """
        if self.id_manager is None:
            return [], [self.name]
        return [self.name], []

    def getElementaryExpression(self, name):
        """

        :return: an elementary expression from its name if it appears in the
            expression. None otherwise.
        :rtype: biogeme.Expression
        """
        if self.name == name:
            return self

        return None

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
        if self.name in names:
            if prefix is not None:
                self.name = f'{prefix}{self.name}'
            if suffix is not None:
                self.name = f'{self.name}{suffix}'


class bioDraws(Elementary):
    """
    Draws for Monte-Carlo integration
    """

    def __init__(self, name, drawType):
        """Constructor

        :param name: name of the random variable with a series of draws.
        :type name: string
        :param drawType: type of draws.
        :type drawType: string
        """
        Elementary.__init__(self, name)
        self.drawType = drawType
        self.drawId = None

    def __str__(self):
        return f'bioDraws("{self.name}", "{self.drawType}")'

    def check_draws(self):
        """List of draws defined outside of 'MonteCarlo'

        :return: List of names of variables
        :rtype: list(str)
        """
        return {self.name}

    def setIdManager(self, id_manager=None):
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may nee to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """
        self.id_manager = id_manager
        if id_manager is None:
            self.elementaryIndex = None
            self.drawId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[
            self.name
        ]
        self.drawId = self.id_manager.draws.indices[self.name]

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }, preceeded by a comma
            3. the name of the draws
            4. the unique ID (preceeded by a comma),
            5. the draw ID (preceeded by a comma).

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

        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            draw
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary '
                f'expression {self.name}.'
            )
            raise excep.biogemeError(error_msg)
        if self.drawId is None:
            error_msg = f'No id has been defined for draw {self.name}.'
            raise excep.biogemeError(error_msg)
        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += f'"{self.name}",{self.elementaryIndex},{self.drawId}'
        return [signature.encode()]

    def dictOfDraws(self):
        """Recursively extract the random variables
        (draws for Monte-Carlo).  Overloads the generic function.
        appearing in the expression, and store them in a dictionary.

        :return: dict where the keys are the random variables and the
                 elements the type of draws. Here, contains only one element.
        :rtype: dict(string:string)
        """
        return {self.name: self.drawType}


class Numeric(Expression):
    """
    Numerical expression for a simple number
    """

    def __init__(self, value):
        """Constructor

        :param value: numerical value
        :type value: float
        """
        Expression.__init__(self)
        self.value = float(value)  #: numeric value

    def __str__(self):
        return '`' + str(self.value) + '`'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.value

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the value, preceeded by a comma.

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
        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += f',{self.value}'
        return [signature.encode()]


class Variable(Elementary):
    """Explanatory variable

    This represents the explanatory variables of the choice
    model. Typically, they come from the data set.
    """

    def __init__(self, name):
        """Constructor

        :param name: name of the variable.
        :type name: string
        """
        Elementary.__init__(self, name)
        # Index of the variable
        self.variableId = None

    def check_panel_trajectory(self):
        """List of variables defined outside of 'PanelLikelihoodTrajectory'

        :return: List of names of variables
        :rtype: list(str)
        """
        return {self.name}

    def getValue(self):
        """The evaluation of a Variable requires a database. Therefore, this
            function triggers an exception.

        :raise biogemeError: each time the function is calles

        """
        error_msg = (
            f'Evaluating Variable {self.name} requires a database. Use the '
            f'function getValue_c instead.'
        )
        raise excep.biogemeError(error_msg)

    def setIdManager(self, id_manager=None):
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may need to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """

        self.id_manager = id_manager
        if id_manager is None:
            self.elementaryIndex = None
            self.variableId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[
            self.name
        ]
        self.variableId = self.id_manager.variables.indices[self.name]

    def dictOfVariables(self):
        """Recursively extract the variables appearing in the expression, and
        store them in a dictionary.

        Overload the generic function.

        :return: returns a dict with the variables appearing in the
               expression the keys being their names.
               Here, it contains only one element.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        return {self.name: self}

    def audit(self, database=None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)

        :raise biogemeError: if no database is provided.

        :raise biogemeError: if the name of the variable does not appear
            in the database.
        """
        listOfErrors = []
        listOfWarnings = []
        if database is None:
            raise excep.biogemeError(
                'The database must be provided to audit the variable.'
            )

        if self.name not in database.data.columns:
            theError = f'Variable {self.name} not found in the database.'
            listOfErrors.append(theError)
        return listOfErrors, listOfWarnings

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the name of the variable,
            4. the unique ID, preceeded by a comma.
            5. the variabvle ID, preceeded by a comma.

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

        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            variable
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary expression '
                f'{self.name}.'
            )
            raise excep.biogemeError(error_msg)
        if self.variableId is None:
            error_msg = f'No id has been defined for variable {self.name}.'
            raise excep.biogemeError(error_msg)
        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += f'"{self.name}",{self.elementaryIndex},{self.variableId}'
        return [signature.encode()]


class DefineVariable(Variable):
    """Expression that defines a new variable and add a column in the database.

    This expression allows the use to define a new variable that
    will be added to the database. It avoids that it is
    recalculated each time it is needed.
    """

    def __init__(self, name, expression, database):
        """Constructor

        :param name: name of the variable.
        :type name: string
        :param expression: formula that defines the variable
        :param type:  biogeme.expressions.Expression
        :param database: object identifying the database.
        :type database: biogeme.database.Database

        :raise biogemeError: if the expression is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        raise excep.biogemeError(
            'This expression is obsolete. Use the same function in the '
            'database object. Replace "new_var = DefineVariable(\'NEW_VAR\','
            ' expression, database)" by  "new_var = database.DefineVariable'
            '(\'NEW_VAR\', expression)"'
        )


class RandomVariable(Elementary):
    """
    Random variable for numerical integration
    """

    def __init__(self, name):
        """Constructor

        :param name: name of the random variable involved in the integration.
        :type name: string.
        """
        Elementary.__init__(self, name)
        # Index of the random variable
        self.rvId = None

    def check_rv(self):
        """List of random variables defined outside of 'Integrate'

        :return: List of names of variables
        :rtype: list(str)
        """
        return {self.name}

    def setIdManager(self, id_manager=None):
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may nee to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """
        self.id_manager = id_manager
        if id_manager is None:
            self.elementaryIndex = None
            self.rvId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[
            self.name
        ]
        self.rvId = self.id_manager.random_variables.indices[self.name]

    def dictOfRandomVariables(self):
        """Recursively extract the random variables appearing in
        the expression, and store them in a dictionary.

        Overloads the generic function.

        :return: returns a dict with the random variables appearing in
                 the expression the keys being their names.

        :rtype: dict(string:biogeme.expressions.Expression)
        """
        return {self.name: self}

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the name of the random variable,
            4. the unique ID, preceeded by a comma,
            5. the ID of the random variable.

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

        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            random variable
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary '
                f'expression {self.name}.'
            )
            raise excep.biogemeError(error_msg)
        if self.rvId is None:
            error_msg = (
                f'No id has been defined for random variable {self.name}.'
            )
            raise excep.biogemeError(error_msg)

        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += f'"{self.name}",{self.elementaryIndex},{self.rvId}'
        return [signature.encode()]


class Beta(Elementary):
    """
    Unknown parameters to be estimated from data.
    """

    def __init__(self, name, value, lowerbound, upperbound, status):
        """Constructor

        :param name: name of the parameter.
        :type name: string
        :param value: default value.
        :type value: float
        :param lowerbound: if different from None, imposes a lower
          bound on the value of the parameter during the optimization.
        :type lowerbound: float
        :param upperbound: if different from None, imposes an upper
          bound on the value of the parameter during the optimization.
        :type upperbound: float
        :param status: if different from 0, the parameter is fixed to
          its default value, and not modified by the optimization algorithm.
        :type status: int

        :raise biogemeError: if the first parameter is not a str.

        :raise biogemeError: if the second parameter is not a int or a float.
        """

        if not isinstance(value, (int, float)):
            error_msg = (
                f'The second parameter for {name} must be '
                f'a float and not a {type(value)}: {value}'
            )
            raise excep.biogemeError(error_msg)
        if not isinstance(name, str):
            error_msg = (
                f'The first parameter must be a string and '
                f'not a {type(name)}: {name}'
            )
            raise excep.biogemeError(error_msg)
        Elementary.__init__(self, name)
        self.initValue = value
        self.lb = lowerbound
        self.ub = upperbound
        self.status = status
        self.betaId = None

    def setIdManager(self, id_manager=None):
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may nee to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """
        self.id_manager = id_manager
        if id_manager is None:
            self.elementaryIndex = None
            self.betaId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[
            self.name
        ]
        if self.status != 0:
            self.betaId = self.id_manager.fixed_betas.indices[self.name]
        else:
            self.betaId = self.id_manager.free_betas.indices[self.name]

    def __str__(self):
        if self.status == 0:
            return f'{self.name}(init={self.initValue})'
        return f'{self.name}(fixed={self.initValue})'

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
        if self.name in beta_values:
            self.initValue = beta_values[self.name]
            self.status = 1
            if prefix is not None:
                self.name = f'{prefix}{self.name}'
            if suffix is not None:
                self.name = f'{self.name}{suffix}'

    def setOfBetas(self, free=True, fixed=False):
        """Extract the set of parameters from the expression. Overload the
        generic function.

        :param free: if True, the free parameters are included. Default: True.
        :type free: bool
        :param fixed: if True, the fixed parameters are included.
            Default: False.
        :type fixed: bool
        :return: returns a set with the beta parameters appearing in the
            expression.
        :rtype: set(biogeme.expressions.Expression)

        """
        if fixed and self.status != 0:
            return set([self.name])

        if free and self.status == 0:
            return set([self.name])

        return set()

    def dictOfBetas(self, free=True, fixed=False):
        """Extract the set of parameters from the expression. Overload the
        generic function.

        :param free: if True, the free parameters are included. Default: True.
        :type free: bool
        :param fixed: if True, the fixed parameters are included.
             Default: False.
        :type fixed: bool

        :return: a dict with the beta parameters appearing in the expression,
                 the keys being the names of the parameters.
        :rtype: dict(string:biogeme.expressions.Expression)
        """
        if fixed and self.status != 0:
            return {self.name: self}

        if free and self.status == 0:
            return {self.name: self}

        return {}

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.initValue

    def changeInitValues(self, betas):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        if self.name in betas:
            self.initValue = betas[self.name]

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the name of the parameter,
            4. the status between [ ]
            5. the unique ID,  preceeded by a comma
            6. the beta ID,  preceeded by a comma


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

        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.biogemeError: if no id has been defined for
            parameter
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary '
                f'expression {self.name}.'
            )
            raise excep.biogemeError(error_msg)
        if self.betaId is None:
            raise excep.biogemeError(
                f'No id has been defined for parameter {self.name}.'
            )

        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += (
            f'"{self.name}"[{self.status}],'
            f'{self.elementaryIndex},{self.betaId}'
        )
        return [signature.encode()]


class LogLogit(Expression):
    """Expression capturing the logit formula.

    It contains one formula for the target alternative, a dict of
    formula for the availabilities and a dict of formulas for the
    utilities

    """

    def __init__(self, util, av, choice):
        """Constructor

        :param util: dictionary where the keys are the identifiers of
                     the alternatives, and the elements are objects
                     defining the utility functions.

        :type util: dict(int:biogeme.expressions.Expression)

        :param av: dictionary where the keys are the identifiers of
                   the alternatives, and the elements are object of
                   type biogeme.expressions.Expression defining the
                   availability conditions. If av is None, all the
                   alternatives are assumed to be always available

        :type av: dict(int:biogeme.expressions.Expression)

        :param choice: formula to obtain the alternative for which the
                       logit probability must be calculated.
        :type choice: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        Expression.__init__(self)
        self.util = {}  #: dict of utility functions
        for i, e in util.items():
            if isNumeric(e):
                self.util[i] = Numeric(e)
            else:
                if not isinstance(e, Expression):
                    raise excep.biogemeError(
                        f'This is not a valid expression: {e}'
                    )
                self.util[i] = e
        self.av = {}  #: dict of availability formulas
        if av is None:
            self.av = {k: Numeric(1) for k, v in util.items()}
        else:
            for i, e in av.items():
                if isNumeric(e):
                    self.av[i] = Numeric(e)
                else:
                    if not isinstance(e, Expression):
                        raise excep.biogemeError(
                            f'This is not a valid expression: {e}'
                        )
                    self.av[i] = e
        if isNumeric(choice):
            self.choice = Numeric(choice)
            """expression for the chosen alternative"""
        else:
            if not isinstance(choice, Expression):
                raise excep.biogemeError(
                    f'This is not a valid expression: {choice}'
                )
            self.choice = choice

        self.children.append(self.choice)
        for i, e in self.util.items():
            self.children.append(e)
        for i, e in self.av.items():
            self.children.append(e)

    def audit(self, database=None):
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)

        """
        listOfErrors = []
        listOfWarnings = []
        for e in self.children:
            err, war = e.audit(database)
            listOfErrors += err
            listOfWarnings += war

        if self.util.keys() != self.av.keys():
            theError = (
                'Incompatible list of alternatives in logit expression. '
            )
            consistent = False
            myset = self.util.keys() - self.av.keys()
            if myset:
                mysetContent = ', '.join(f'{str(k)} ' for k in myset)
                theError += (
                    'Id(s) used for utilities and not for ' 'availabilities: '
                ) + mysetContent
            myset = self.av.keys() - self.util.keys()
            if myset:
                mysetContent = ', '.join(f'{str(k)} ' for k in myset)
                theError += (
                    ' Id(s) used for availabilities and not ' 'for utilities: '
                ) + mysetContent
            listOfErrors.append(theError)
        else:
            consistent = True
        listOfAlternatives = list(self.util)
        if database is None:
            choices = np.array([self.choice.getValue_c()])
        else:
            choices = database.valuesFromDatabase(self.choice)
        correctChoices = np.isin(choices, listOfAlternatives)
        indexOfIncorrectChoices = np.argwhere(~correctChoices)
        if indexOfIncorrectChoices.any():
            incorrectChoices = choices[indexOfIncorrectChoices]
            content = '-'.join(
                '{}[{}]'.format(*t)
                for t in zip(indexOfIncorrectChoices, incorrectChoices)
            )
            truncate = 100
            if len(content) > truncate:
                content = f'{content[:truncate]}...'
            theError = (
                f'The choice variable [{self.choice}] does not '
                f'correspond to a valid alternative for the '
                f'following observations (rownumber[choice]): '
            ) + content
            listOfErrors.append(theError)

        if consistent:
            if database is None:
                value_choice = self.choice.getValue_c()
                if not value_choice in self.av.keys():
                    theError = (
                        f'The chosen alternative [{value_choice}] '
                        f'is not available'
                    )
                    listOfWarnings.append(theError)
            else:
                choiceAvailability = database.checkAvailabilityOfChosenAlt(
                    self.av, self.choice
                )
                indexOfUnavailableChoices = np.where(~choiceAvailability)[0]
                if indexOfUnavailableChoices.size > 0:
                    incorrectChoices = choices[indexOfUnavailableChoices]
                    content = '-'.join(
                        '{}[{}]'.format(*t)
                        for t in zip(
                            indexOfUnavailableChoices, incorrectChoices
                        )
                    )
                    truncate = 100
                    if len(content) > truncate:
                        content = f'{content[:truncate]}...'
                    theError = (
                        f'The chosen alternative [{self.choice}] '
                        f'is not available for the following '
                        f'observations (rownumber[choice]): '
                    ) + content
                    listOfWarnings.append(theError)

        return listOfErrors, listOfWarnings

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise biogemeError: if the chosen alternative does not correspond
            to any of the utility functions

        :raise biogemeError: if the chosen alternative does not correspond
            to any of entry in the availability condition

        """
        choice = int(self.choice.getValue())
        if choice not in self.util:
            error_msg = (
                f'Alternative {choice} does not appear in the list '
                f'of utility functions: {self.util.keys()}'
            )
            raise excep.biogemeError(error_msg)
        if choice not in self.av:
            error_msg = (
                f'Alternative {choice} does not appear in the list '
                f'of availabilities: {self.av.keys()}'
            )
            raise excep.biogemeError(error_msg)
        if self.av[choice].getValue() == 0.0:
            return -np.log(0)
        Vchosen = self.util[choice].getValue()
        denom = 0.0
        for i, V in self.util.items():
            if self.av[i].getValue() != 0.0:
                denom += np.exp(V.getValue() - Vchosen)
        return -np.log(denom)

    def __str__(self):
        s = self.getClassName()
        s += f'[choice={self.choice}]'
        s += 'U=('
        first = True
        for i, e in self.util.items():
            if first:
                s += f'{int(i)}:{e}'
                first = False
            else:
                s += f', {int(i)}:{e}'
        s += ')'
        s += 'av=('
        first = True
        for i, e in self.av.items():
            if first:
                s += f'{int(i)}:{e}'
                first = False
            else:
                s += f', {int(i)}:{e}'
        s += ')'
        return s

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of all the children expressions,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the number of alternatives between ( )
            5. the id of the expression for the chosen alternative, preceeded
               by a comma.
            6. for each alternative, separated by commas:

                 a. the number of the alternative, as defined by the user,
                 b. the id of the expression for the utility,
                 c. the id of the expression for the availability condition.

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
        listOfSignatures = []
        for e in self.children:
            listOfSignatures += e.getSignature()
        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += f'({len(self.util)})'
        signature += f',{id(self.choice)}'
        for i, e in self.util.items():
            signature += f',{i},{id(e)},{id(self.av[i])}'
        listOfSignatures += [signature.encode()]
        return listOfSignatures


class _bioLogLogit(LogLogit):
    """log of logit formula

    This expression captures the logarithm of the logit formula. It
    contains one formula for the target alternative, a dict of formula
    for the availabilities and a dict of formulas for the utilities It
    uses only the C++ implementation.
    """

    def __init__(self, util, av, choice):
        """Constructor

        :param util: dictionary where the keys are the identifiers of
                     the alternatives, and the elements are objects
                     defining the utility functions.

        :type util: dict(int:biogeme.expressions.Expression)

        :param av: dictionary where the keys are the identifiers of
                   the alternatives, and the elements are object of
                   type biogeme.expressions.Expression defining the
                   availability conditions. If av is None, all the
                   alternatives are assumed to be always available

        :type av: dict(int:biogeme.expressions.Expression)

        :param choice: formula to obtain the alternative for which the
                       logit probability must be calculated.
        :type choice: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        super().__init__(util, av, choice)


class _bioLogLogitFullChoiceSet(LogLogit):
    """This expression captures the logarithm of the logit formula, where
    all alternatives are supposed to be always available.

       It contains one formula for the target alternative and a dict of
       formulas for the utilities. It uses only the C++ implementation.

    """

    def __init__(self, util, av, choice):
        """Constructor

        :param util: dictionary where the keys are the identifiers of
                     the alternatives, and the elements are objects
                     defining the utility functions.

        :type util: dict(int:biogeme.expressions.Expression)

        :param av: dictionary where the keys are the identifiers of
                   the alternatives, and the elements are object of
                   type biogeme.expressions.Expression defining the
                   availability conditions. If av is None, all the
                   alternatives are assumed to be always available

        :type av: dict(int:biogeme.expressions.Expression)

        :param choice: formula to obtain the alternative for which the
                       logit probability must be calculated.
        :type choice: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        super().__init__(util, av, choice)


class bioMultSum(Expression):
    """This expression returns the sum of several other expressions.

    It is a generalization of 'Plus' for more than two terms
    """

    def __init__(self, listOfExpressions):
        """Constructor

        :param listOfExpressions: list of objects representing the
                                     terms of the sum.

        :type listOfExpressions: list(biogeme.expressions.Expression)

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        Expression.__init__(self)
        if isinstance(listOfExpressions, dict):
            for e in listOfExpressions.values():
                if isNumeric(e):
                    theExpression = Numeric(e)
                    self.children.append(theExpression)
                else:
                    if not isinstance(e, Expression):
                        raise excep.biogemeError(
                            f'This is not a valid expression: {e}'
                        )
                    self.children.append(e)
        elif isinstance(listOfExpressions, list):
            for e in listOfExpressions:
                if isNumeric(e):
                    theExpression = Numeric(e)
                    self.children.append(theExpression)
                else:
                    if not isinstance(e, Expression):
                        raise excep.biogemeError(
                            f'This is not a valid expression: {e}'
                        )
                    self.children.append(e)
        else:
            raise excep.biogemeError(
                'Argument of bioMultSum must be a dict or a list.'
            )

    def getValue(self):

        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        result = 0.0
        for e in self.children:
            result += e.getValue()
        return result

    def __str__(self):
        s = 'bioMultSum(' + ', '.join([f'{e}' for e in self.children]) + ')'
        return s


class Elem(Expression):
    """This returns the element of a dictionary. The key is evaluated
    from an expression.
    """

    def __init__(self, dictOfExpressions, keyExpression):
        """Constructor

        :param dictOfExpressions: dict of objects with numerical keys.
        :type dictOfExpressions: dict(int: biogeme.expressions.Expression)

        :param keyExpression: object providing the key of the element
                              to be evaluated.
        :type keyExpression: biogeme.expressions.Expression

        :raise biogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        Expression.__init__(self)

        if isinstance(keyExpression, bool):
            self.keyExpression = Numeric(1) if keyExpression else Numeric(0)
        elif isNumeric(keyExpression):
            self.keyExpression = Numeric(keyExpression)
        else:
            if not isinstance(keyExpression, Expression):
                raise excep.biogemeError(
                    f'This is not a valid expression: {keyExpression}'
                )
            self.keyExpression = keyExpression  #: expression for the key
        self.children.append(self.keyExpression)

        self.dictOfExpressions = {}  #: dict of expressions
        for k, v in dictOfExpressions.items():
            if isNumeric(v):
                self.dictOfExpressions[k] = Numeric(v)
            else:
                if not isinstance(v, Expression):
                    raise excep.biogemeError(
                        f'This is not a valid expression: {v}'
                    )
                self.dictOfExpressions[k] = v
            self.children.append(self.dictOfExpressions[k])

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise biogemeError: if the calcuated key is not present in
            the dictionary.
        """
        key = int(self.keyExpression.getValue())
        if key in self.dictOfExpressions:
            return self.dictOfExpressions[key].getValue()

        error_msg = (
            f'Key {key} is not present in the dictionary. '
            f'Available keys: {self.dictOfExpressions.keys()}'
        )
        raise excep.biogemeError(error_msg)

    def __str__(self):
        s = '{{'
        first = True
        for k, v in self.dictOfExpressions.items():
            if first:
                s += f'{k}:{v}'
                first = False
            else:
                s += f', {k}:{v}'
        s += f'}}[{self.keyExpression}]'
        return s

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signature of the expression defining the key
            2. the signatures of all the children expressions,
            3. the name of the expression between < >
            4. the id of the expression between { }
            5. the number of elements between ( )
            6. the id of the expression defining the key
            7. for each element: the value of the key and the id
               of the expression, separated by commas.

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
        listOfSignatures = []
        listOfSignatures += self.keyExpression.getSignature()
        for i, e in self.dictOfExpressions.items():
            listOfSignatures += e.getSignature()
        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += f'({len(self.dictOfExpressions)})'
        signature += f',{id(self.keyExpression)}'
        for i, e in self.dictOfExpressions.items():
            signature += f',{i},{id(e)}'
        listOfSignatures += [signature.encode()]
        return listOfSignatures


class bioLinearUtility(Expression):
    """When the utility function is linear, it is expressed as a list of
    terms, where a parameter multiplies a variable.
    """

    def __init__(self, listOfTerms):
        """Constructor

        :param listOfTerms: a list of tuple. Each tuple contains first
             a beta parameter, second the name of a variable.
        :type listOfTerms: list(biogeme.expressions.Expression,
            biogeme.expressions.Expression)

        :raises biogeme.exceptions.biogemeError: if the object is not
                        a list of tuples (parameter, variable)

        """
        Expression.__init__(self)

        theError = ""
        first = True

        for b, v in listOfTerms:
            if not isinstance(b, Beta):
                if first:
                    theError += (
                        'Each element of the bioLinearUtility '
                        'must be a tuple (parameter, variable). '
                    )
                    first = False
                theError += f' Expression {b} is not a parameter.'
            if not isinstance(v, Variable):
                if first:
                    theError += (
                        'Each element of the list should be '
                        'a tuple (parameter, variable).'
                    )
                    first = False
                theError += f' Expression {v} is not a variable.'
        if not first:
            raise excep.biogemeError(theError)

        self.betas, self.variables = zip(*listOfTerms)

        self.betas = list(self.betas)  #: list of parameters

        self.variables = list(self.variables)  #: list of variables

        self.listOfTerms = list(zip(self.betas, self.variables))
        """ List of terms """

        self.children += self.betas + self.variables

    def __str__(self):
        return ' + '.join([f'{b} * {x}' for b, x in self.listOfTerms])

    def setOfBetas(self, free=True, fixed=False):
        """
        Extract the set of parameters from the expression.

        :param free: if True, the free parameters are included. Default: True.
        :type free: bool
        :param fixed: if True, the fixed parameters are included.
             Default: False.
        :type fixed: bool
        :return: returns a set with the beta parameters appearing in the
            expression.
        :rtype: set(biogeme.expressions.Expression)

        """
        if free:
            return set(self.betas)

        return set()

    def dictOfBetas(self, free=True, fixed=False):
        """
        Extract the set of parameters from the expression.

        :param free: if True, the free parameters are included. Default: True.
        :type free: bool
        :param fixed: if True, the fixed parameters are included.
            Default: False.
        :type fixed: bool

        :return: a dict with the beta parameters appearing
             in the expression, the keys being the names of the parameters.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        freenames = {x.name: x for x in self.betas if x.status == 0}
        fixednames = {x.name: x for x in self.betas if x.status != 0}
        if free and fixed:
            allnames = {**freenames, **fixednames}
            return allnames
        if free:
            return freenames
        if fixed:
            return fixednames
        return {}

    def dictOfVariables(self):
        """Recursively extract the variables appearing in the expression, and
        store them in a dictionary.

        Overloads the generic function.

        :return: returns a dict with the variables appearing
                 in the expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        return {x.name: x for x in self.variables}

    def dictOfRandomVariables(self):
        """Recursively extract the random variables appearing in the
        expression, and store them in a dictionary.

        :return: returns a dict with the random variables appearing in
                 the expression the keys being their names.

        :rtype: dict(string:biogeme.expressions.Expression)

        """
        return {}

    def dictOfDraws(self):
        """Recursively extract the random variables
        (draws for Monte-Carlo).  Overloads the generic function.
        appearing in the expression, and store them in a dictionary.

        :return: dict where the keys are the random variables and
             the elements the type of draws. Here, returns an empty dict.
        :rtype: dict(string:string)
        """
        return {}

    def getSignature(self):
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the signatures of all the children expressions,
            2. the name of the expression between < >
            3. the id of the expression between { }
            4. the number of terms in the utility ( )
            5. for each term:

                a. the id of the beta parameter
                b. the unique id of the beta parameter
                c. the name of the parameter
                d. the id of the variable
                e. the unique id of the variable
                f. the name of the variable

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
        listOfSignatures = []
        for e in self.children:
            listOfSignatures += e.getSignature()
        signature = f'<{self.getClassName()}>'
        signature += f'{{{id(self)}}}'
        signature += f'({len(self.listOfTerms)})'
        for b, v in self.listOfTerms:
            signature += (
                f',{id(b)},{b.elementaryIndex},{b.name},'
                f'{id(v)},{v.elementaryIndex},{v.name}'
            )
        listOfSignatures += [signature.encode()]
        return listOfSignatures
