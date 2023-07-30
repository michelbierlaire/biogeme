"""Defines the interface for a catalog of expressions that may be
considered in a specification

:author: Michel Bierlaire
:date: Sun Feb  5 15:34:56 2023

"""
import logging
import abc
from typing import NamedTuple
import biogeme.expressions as ex
import biogeme.exceptions as excep
from biogeme.configuration import (
    SEPARATOR,
    SELECTION_SEPARATOR,
)

logger = logging.getLogger(__name__)


class NamedExpression(NamedTuple):
    name: str
    expression: ex.Expression


class CatalogItem(NamedTuple):
    catalog_name: str
    item_index: int
    item_name: str


class MultipleExpression(ex.Expression, metaclass=abc.ABCMeta):
    """Interface for catalog of expressions that are interchangeable. Only one of
    them defines the specification. They are designed to be
    modified algorithmically.
    """

    def __init__(self, name):
        if SEPARATOR in name or SELECTION_SEPARATOR in name:
            error_msg = (
                f'Invalid name: {name}. Cannot contain characters '
                f'{SELECTION_SEPARATOR} or {SELECTION_SEPARATOR}'
            )
            raise excep.BiogemeError(error_msg)
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def selected(self):
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: tuple(str, biogeme.expressions.Expression)
        """

    @abc.abstractmethod
    def get_iterator(self):
        """Returns an iterator on NamedExpression"""

    def catalog_size(self):
        """Provide the size of the catalog

        :return: number of expressions in the catalog
        :rtype: int
        """
        the_iterator = self.get_iterator()

        return len(list(the_iterator))

    def selected_name(self):
        """Obtain the name of the selection

        :return: the name of the selected expression
        :rtype: str
        """
        name, _ = self.selected()
        return name

    def selected_expression(self):
        """Obtain the selected expression

        :return: the selected expression
        :rtype: biogeme.expressions.Expression
        """
        _, the_expression = self.selected()
        return the_expression

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        _, expr = self.selected()
        return expr.getValue()

    def get_id(self):
        """Retrieve the id of the expression used in the signature

        :return: id of the object
        :rtype: int
        """
        _, expr = self.selected()
        return expr.get_id()

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
        _, expr = self.selected()
        return expr.getSignature()

    def get_children(self):
        """Retrieve the list of children

        :return: list of children
        :rtype: list(Expression)
        """
        _, expr = self.selected()
        return expr.get_children()

    def __str__(self):
        named_expression = self.selected()
        return f'[{self.name}: {named_expression.name}]{named_expression.expression}'

    def setIdManager(self, id_manager):
        """The ID manager contains the IDs of the elementary expressions.

        It is externally created, as it may nee to coordinate the
        numbering of several expressions. It is stored only in the
        expressions of type Elementary.

        :param id_manager: ID manager to be propagated to the
            elementary expressions. If None, all the IDs are set to None.
        :type id_manager: class IdManager
        """
        _, expr = self.selected()
        expr.setIdManager(id_manager)

    def check_panel_trajectory(self):
        """Set of variables defined outside of 'PanelLikelihoodTrajectory'

        :return: List of names of variables
        :rtype: set(str)
        """
        _, expr = self.selected()
        return expr.check_panel_trajectory()

    def check_draws(self):
        """Set of draws defined outside of 'MonteCarlo'

        :return: List of names of variables
        :rtype: set(str)
        """
        _, expr = self.selected()
        return expr.check_draws()

    def check_rv(self):
        """Set of random variables defined outside of 'Integrate'

        :return: List of names of variables
        :rtype: set(str)
        """
        _, expr = self.selected()
        return expr.check_rv()

    def getStatusIdManager(self):
        """Check the elementary expressions that are associated with
        an ID manager.

        :return: two sets of elementary expressions, those with and
            without an ID manager.
        :rtype: tuple(set(str), set(str))
        """
        _, expr = self.selected()
        return expr.getStatusIdManager()

    def getElementaryExpression(self, name):
        """Return: an elementary expression from its name if it appears in the
        expression.

        :param name: name of the elementary expression.
        :type name: string

        :return: the expression if it exists. None otherwise.
        :rtype: biogeme.expressions.Expression
        """
        _, expr = self.selected()
        return expr.getElementaryExpression(name)

    def set_of_elementary_expression(self, the_type):
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression

        :return: returns a set with the names of the elementary expressions
        :rtype: set(string.Expression)

        """
        _, expr = self.selected()
        return expr.set_of_elementary_expression(the_type)

    def dict_of_elementary_expression(self, the_type):
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression

        :return: returns a dict with the variables appearing in the
               expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        _, expr = self.selected()
        return expr.dict_of_elementary_expression(the_type)

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
        _, expr = self.selected()
        return expr.rename_elementary(names, prefix, suffix)

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
        _, expr = self.selected()
        return expr.fix_betas(beta_values, prefix, suffix)

    def embedExpression(self, t):
        """Check if the expression contains an expression of type t.

        Typically, this would be used to check that a MonteCarlo
        expression contains a bioDraws expression.

        :return: True if the expression contains an expression of type t.
        :rtype: bool

        """
        _, expr = self.selected()
        return expr.embedExpression(t)

    def countPanelTrajectoryExpressions(self):
        """Count the number of times the PanelLikelihoodTrajectory
        is used in the formula. It should trigger an error if it
        is used more than once.

        :return: number of times the PanelLikelihoodTrajectory
            is used in the formula
        :rtype: int
        """
        _, expr = self.selected()
        return expr.countPanelTrajectoryExpressions()

    def change_init_values(self, betas):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """
        _, expr = self.selected()
        expr.change_init_values(betas)
