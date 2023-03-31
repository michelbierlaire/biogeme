"""Defines the interface for a catalog of expressions that may be
considered in a specification

:author: Michel Bierlaire
:date: Sun Feb  5 15:34:56 2023

"""
import abc
from collections import namedtuple
import biogeme.exceptions as excep
import biogeme.expressions as ex


NamedExpression = namedtuple('NamedExpression', 'name expression')

CatalogItem = namedtuple('CatalogItem', 'catalog_name item_index item_name')


class MultipleExpression(ex.Expression, metaclass=abc.ABCMeta):
    """Interface for catalog of expressions that are interchangeable. Only one of
    them defines the specification. They are designed to be
    modified algorithmically.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name
        
    @abc.abstractmethod
    def selected(self):
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: tuple(str, biogeme.expressions.Expression)
        """

    @abc.abstractmethod
    def increment_selection(self):
        """Increment recursively the selection of multiple
        expressions.

        :return: True if the increment has been implemented
        :rtype: bool
        """

    @abc.abstractmethod
    def increment_catalog(self, catalog_name, step):
        """Increment the selection of a specific catalog

        :param catalog_name: name of the catalog to change
        :type catalog_name: str

        :param step: number of increments to apply. Can be negative.
        :type step: int
        """

    @abc.abstractmethod
    def configure_catalogs(self, configuration):
        """Select the items in each catalog corresponding to the
            requested configuration

        :param configuration: a dictionary such that the keys are the catalog
            names, and the values are the selected specification in
            the Catalog
        :type configuration: dict(str: str)

        """

    @abc.abstractmethod
    def get_iterator(self):
        """Returns an iterator on NamedExpression"""


    @abc.abstractmethod
    def reset_this_expression_selection(self):
        """In this group of expressions, select the first one"""


    @abc.abstractmethod
    def current_configuration(self):
        """Obtain the current configuration of an expression with Catalog's

        :return: a dictionary such that the keys are the catalog
            names, and the values are the selected specification in
            the Catalog

        :rtype: dict(str: str)
        """

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

    def dict_of_catalogs(self, ignore_synchronized=False):
        """Returns a dict with all catalogs in the expression

        :return: dict with all the catalogs
        """
        result = {}
        for _, expr in self.get_iterator():
            a_dict = expr.dict_of_catalogs()
            for key, the_catalog in a_dict.items():
                if key in result:
                    error_msg = (
                        f'Catalog {key} cannot appear twice in the same '
                        f'expression. Use different names.'
                    )
                    raise excep.biogemeError(error_msg)
                result[key] = the_catalog
        result[self.name] = self
        return result

    def number_of_multiple_expressions(self):
        """Reports the number of multiple expressions available through the iterator

        :return: number of multiple expressions. It just gives an
            upper bound if some catalogs are synchronized
        :rtype: int

        """
        the_iterator = self.get_iterator()
        total = 0
        for _, expr in the_iterator:
            total = total + expr.number_of_multiple_expressions()
        return total

    def reset_expression_selection(self):
        """In each group of expressions, select the first one"""
        self.reset_this_expression_selection()
        for _, expression in self.get_iterator():
            expression.reset_expression_selection()

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

    def changeInitValues(self, betas):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """
        _, expr = self.selected()
        expr.changeInitValues(betas)
