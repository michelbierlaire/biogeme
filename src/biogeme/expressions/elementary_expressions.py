""" Arithmetic expressions accepted by Biogeme: elementary expressions

:author: Michel Bierlaire
:date: Tue Mar  7 18:38:21 2023

"""
import logging
import biogeme.exceptions as excep
from .base_expressions import Expression
from .elementary_types import TypeOfElementaryExpression
from .numeric_tools import validate, MAX_VALUE

logger = logging.getLogger(__name__)


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

    def number_of_multiple_expressions(self):
        """Count the number of "parallel" expressions

        :return: the number of expressions
        :rtype: int
        """
        return 1


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
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            draw
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary ' f'expression {self.name}.'
            )
            raise excep.BiogemeError(error_msg)
        if self.drawId is None:
            error_msg = f'No id has been defined for draw {self.name}.'
            raise excep.BiogemeError(error_msg)
        signature = f'<{self.getClassName()}>'
        signature += f'{{{self.get_id()}}}'
        signature += f'"{self.name}",{self.elementaryIndex},{self.drawId}'
        return [signature.encode()]

    def dict_of_elementary_expression(self, the_type):
        """Extract a dict with all elementary expressions of a dpecific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """
        if the_type == TypeOfElementaryExpression.DRAWS:
            return {self.name: self.drawType}
        return {}


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
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
        self.variableId = self.id_manager.variables.indices[self.name]

    def dict_of_elementary_expression(self, the_type):
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """
        if the_type == TypeOfElementaryExpression.VARIABLE:
            return {self.name: self}
        return {}

    def audit(self, database=None):
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
            raise excep.BiogemeError(
                'The database must be provided to audit the variable.'
            )

        if self.name not in database.data.columns:
            the_error = f'Variable {self.name} not found in the database.'
            list_of_errors.append(the_error)
        return list_of_errors, list_of_warnings

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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            variable
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary expression ' f'{self.name}.'
            )
            raise excep.BiogemeError(error_msg)
        if self.variableId is None:
            error_msg = f'No id has been defined for variable {self.name}.'
            raise excep.BiogemeError(error_msg)
        signature = f'<{self.getClassName()}>'
        signature += f'{{{self.get_id()}}}'
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
        :type expression:  biogeme.expressions.Expression
        :param database: object identifying the database.
        :type database: biogeme.database.Database

        :raise BiogemeError: if the expression is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.
        """
        raise excep.BiogemeError(
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
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
        self.rvId = self.id_manager.random_variables.indices[self.name]

    def dict_of_elementary_expression(self, the_type):
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """
        if the_type == TypeOfElementaryExpression.RANDOM_VARIABLE:
            return {self.name: self}
        return {}

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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            random variable
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary ' f'expression {self.name}.'
            )
            raise excep.BiogemeError(error_msg)
        if self.rvId is None:
            error_msg = f'No id has been defined for random variable {self.name}.'
            raise excep.BiogemeError(error_msg)

        signature = f'<{self.getClassName()}>'
        signature += f'{{{self.get_id()}}}'
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

        :raise BiogemeError: if the first parameter is not a str.

        :raise BiogemeError: if the second parameter is not a int or a float.
        """

        if not isinstance(value, (int, float)):
            error_msg = (
                f'The second parameter for {name} must be '
                f'a float and not a {type(value)}: {value}'
            )
            raise excep.BiogemeError(error_msg)
        if not isinstance(name, str):
            error_msg = (
                f'The first parameter must be a string and '
                f'not a {type(name)}: {name}'
            )
            raise excep.BiogemeError(error_msg)
        Elementary.__init__(self, name)
        the_value = validate(value, modify=False)
        self.initValue = the_value
        self.estimated_value = None
        if lowerbound is not None:
            the_lowerbound = validate(lowerbound, modify=False)
            self.lb = the_lowerbound
        else:
            self.lb = -MAX_VALUE
        if upperbound is not None:
            the_upperbound = validate(upperbound, modify=False)
            self.ub = the_upperbound
        else:
            self.ub = MAX_VALUE
        self.status = status
        self.betaId = None

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
            self.betaId = None
            return
        self.elementaryIndex = self.id_manager.elementary_expressions.indices[self.name]
        if self.status != 0:
            self.betaId = self.id_manager.fixed_betas.indices[self.name]
        else:
            self.betaId = self.id_manager.free_betas.indices[self.name]

    def __str__(self):
        return (
            f"Beta('{self.name}', {self.initValue}, {self.lb}, "
            f"{self.ub}, {self.status})"
        )

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

    def dict_of_elementary_expression(self, the_type):
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression

        :return: returns a dict with the variables appearing in the
               expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        if the_type == TypeOfElementaryExpression.BETA:
            return {self.name: self}

        if the_type == TypeOfElementaryExpression.FREE_BETA and self.status == 0:
            return {self.name: self}

        if the_type == TypeOfElementaryExpression.FIXED_BETA and self.status != 0:
            return {self.name: self}

        return {}

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise BiogemeError: if the Beta is not fixed.
        """
        if self.status == 0:
            if self.estimated_value is None:
                error_msg = f'Parameter {self.name} must be estimated from data.'
                raise excep.BiogemeError(error_msg)
            return self.estimated_value
        return self.initValue

    def change_init_values(self, betas):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        if self.name in betas:
            self.initValue = betas[self.name]

    def set_estimated_values(self, betas: dict[str, float]):
        """Set the estimated values of beta

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        """
        if self.name in betas:
            self.estimated_value = betas[self.name]

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

        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            elementary expression
        :raise biogeme.exceptions.BiogemeError: if no id has been defined for
            parameter
        """
        if self.elementaryIndex is None:
            error_msg = (
                f'No id has been defined for elementary ' f'expression {self.name}.'
            )
            raise excep.BiogemeError(error_msg)
        if self.betaId is None:
            raise excep.BiogemeError(
                f'No id has been defined for parameter {self.name}.'
            )

        signature = f'<{self.getClassName()}>'
        signature += f'{{{self.get_id()}}}'
        signature += (
            f'"{self.name}"[{self.status}],' f'{self.elementaryIndex},{self.betaId}'
        )
        return [signature.encode()]
