""" Arithmetic expressions accepted by Biogeme: elementary expressions

:author: Michel Bierlaire
:date: Tue Mar  7 18:38:21 2023

"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from biogeme.exceptions import BiogemeError
from .base_expressions import Expression
from .elementary_types import TypeOfElementaryExpression

if TYPE_CHECKING:
    from biogeme.database import Database

logger = logging.getLogger(__name__)


class Elementary(Expression):
    """Elementary expression.

    It is typically defined by a name appearing in an expression. It
    can be a variable (from the database), or a parameter (fixed or to
    be estimated using maximum likelihood), a random variable for
    numerical integration, or Monte-Carlo integration.

    """

    def __init__(self, name: str):
        """Constructor

        :param name: name of the elementary expression.
        :type name: string

        """
        Expression.__init__(self)
        self.name = name  #: name of the elementary expression

        self.elementaryIndex = None
        """The id should be unique for all elementary expressions
        appearing in a given set of formulas.
        """

    def __str__(self) -> str:
        """string method

        :return: name of the expression
        :rtype: str
        """
        return f"{self.name}"

    def get_status_id_manager(self) -> tuple[list[str], list[str]]:
        """Check the elementary expressions that are associated with
        an ID manager.

        :return: two lists of elementary expressions, those with and
            without an ID manager.
        :rtype: tuple(list(str), list(str))
        """
        if self.id_manager is None:
            return [], [self.name]
        return [self.name], []

    def get_elementary_expression(self, name: str) -> Expression | None:
        """

        :return: an elementary expression from its name if it appears in the
            expression. None otherwise.
        :rtype: biogeme.Expression
        """
        if self.name == name:
            return self

        return None

    def rename_elementary(
        self, names: list[str], prefix: str | None = None, suffix: str | None = None
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
        if self.name in names:
            if prefix is not None:
                self.name = f"{prefix}{self.name}"
            if suffix is not None:
                self.name = f"{self.name}{suffix}"

    def number_of_multiple_expressions(self) -> int:
        """Count the number of "parallel" expressions

        :return: the number of expressions
        :rtype: int
        """
        return 1


class bioDraws(Elementary):
    """
    Draws for Monte-Carlo integration
    """

    def __init__(self, name: str, draw_type: str):
        """Constructor

        :param name: name of the random variable with a series of draws.
        :type name: string
        :param draw_type: type of draws.
        :type draw_type: string
        """
        Elementary.__init__(self, name)
        self.drawType = draw_type
        self.drawId = None

    def __str__(self) -> str:
        return f'bioDraws("{self.name}", "{self.drawType}")'

    def check_draws(self) -> set[str]:
        """Set of draws defined outside of 'MonteCarlo'

        :return: List of names of variables
        :rtype: list(str)
        """
        return {self.name}

    def set_id_manager(self, id_manager: IdManager | None = None):
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
            self.drawId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
        self.drawId = self.id_manager.draws.indices[self.name]

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }, preceded by a comma
            3. the name of the draws
            4. the unique ID (preceded by a comma),
            5. the draw ID (preceded by a comma).

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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            draw
        """
        if self.elementaryIndex is None:
            error_msg = (
                f"No id has been defined for elementary " f"expression {self.name}."
            )
            raise BiogemeError(error_msg)
        if self.drawId is None:
            error_msg = f"No id has been defined for draw {self.name}."
            raise BiogemeError(error_msg)
        signature = f"<{self.get_class_name()}>"
        signature += f"{{{self.get_id()}}}"
        signature += f'"{self.name}",{self.elementaryIndex},{self.drawId}'
        return [signature.encode()]

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str, Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """

        if the_type == TypeOfElementaryExpression.DRAWS:
            # Until version 3.2.13, this function returned the following:
            # return {self.name: self.drawType}
            return {self.name: self}
        return {}


class Variable(Elementary):
    """Explanatory variable

    This represents the explanatory variables of the choice
    model. Typically, they come from the data set.
    """

    def __init__(self, name: str):
        """Constructor

        :param name: name of the variable.
        :type name: string
        """
        Elementary.__init__(self, name)
        # Index of the variable
        self.variableId = None

    def check_panel_trajectory(self) -> set[str]:
        """Set of variables defined outside of 'PanelLikelihoodTrajectory'

        :return: List of names of variables
        :rtype: list(str)
        """
        return {self.name}

    def set_id_manager(self, id_manager: IdManager | None = None):
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
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
        self.variableId = self.id_manager.variables.indices[self.name]

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str:Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """
        if the_type == TypeOfElementaryExpression.VARIABLE:
            return {self.name: self}
        return {}

    def audit(self, database: Database | None = None) -> tuple[list[str], list[str]]:
        """Performs various checks on the expressions.

        :param database: database object
        :type database: biogeme.database.Database

        :return: tuple list_of_errors, list_of_warnings
        :rtype: list(string), list(string)

        :raise BiogemeError: if no database is provided.

        :raise BiogemeError: if the name of the variable does not appear
            in the database.
        """
        list_of_errors = []
        list_of_warnings = []
        if database is None:
            raise BiogemeError("The database must be provided to audit the variable.")

        if self.name not in database.data.columns:
            the_error = f"Variable {self.name} not found in the database."
            list_of_errors.append(the_error)
        return list_of_errors, list_of_warnings

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the name of the variable,
            4. the unique ID, preceded by a comma.
            5. the variable ID, preceded by a comma.

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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            variable
        """
        if self.elementaryIndex is None:
            error_msg = (
                f"No id has been defined for elementary expression " f"{self.name}."
            )
            raise BiogemeError(error_msg)
        if self.variableId is None:
            error_msg = f"No id has been defined for variable {self.name}."
            raise BiogemeError(error_msg)
        signature = f"<{self.get_class_name()}>"
        signature += f"{{{self.get_id()}}}"
        signature += f'"{self.name}",{self.elementaryIndex},{self.variableId}'
        return [signature.encode()]


class DefineVariable(Variable):
    """
    .. warning:: This expression is obsolete.  Replace
         `new_var = DefineVariable('NEW_VAR', expression, database)` by
         `new_var = database.DefineVariable('NEW_VAR', expression)`
    """

    def __init__(self, name: str, expression: Expression, database: Database):
        """Constructor

        :param name: name of the variable.
        :param expression: formula that defines the variable
        :param database: object identifying the database.

        :raise BiogemeError: if the expression is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        raise BiogemeError(
            "This expression is obsolete. Use the same function in the "
            "database object. Replace \"new_var = DefineVariable('NEW_VAR',"
            ' expression, database)" by  "new_var = database.DefineVariable'
            "('NEW_VAR', expression)\""
        )


class RandomVariable(Elementary):
    """
    Random variable for numerical integration
    """

    def __init__(self, name: str):
        """Constructor

        :param name: name of the random variable involved in the integration.
        :type name: string.
        """
        Elementary.__init__(self, name)
        # Index of the random variable
        self.rvId: int | None = None

    def check_rv(self) -> set[str]:
        """Set of random variables defined outside of 'Integrate'

        :return: List of names of variables
        :rtype: list(str)
        """
        return {self.name}

    def set_id_manager(self, id_manager: IdManager | None = None):
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
            self.rvId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
        self.rvId = self.id_manager.random_variables.indices[self.name]

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str:Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """
        if the_type == TypeOfElementaryExpression.RANDOM_VARIABLE:
            return {self.name: self}
        return {}

    def get_signature(self) -> list[bytes]:
        """The signature of a string characterizing an expression.

        This is designed to be communicated to C++, so that the
        expression can be reconstructed in this environment.

        The list contains the following elements:

            1. the name of the expression between < >
            2. the id of the expression between { }
            3. the name of the random variable,
            4. the unique ID, preceded by a comma,
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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            random variable
        """
        if self.elementaryIndex is None:
            error_msg = (
                f"No id has been defined for elementary " f"expression {self.name}."
            )
            raise BiogemeError(error_msg)
        if self.rvId is None:
            error_msg = f"No id has been defined for random variable {self.name}."
            raise BiogemeError(error_msg)

        signature = f"<{self.get_class_name()}>"
        signature += f"{{{self.get_id()}}}"
        signature += f'"{self.name}",{self.elementaryIndex},{self.rvId}'
        return [signature.encode()]

