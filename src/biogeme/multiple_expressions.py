""" Defines a catalog of expressions that may be considered in a specification

:author: Michel Bierlaire
:date: Sun Feb  5 15:34:56 2023

"""

import biogeme.exceptions as excep
from biogeme.expressions import Expression


class MultipleExpressions(Expression):
    """Catalog of expressions that are interchangeable. Only one of
        them defines the specification. They are designed to be
        modified algorithmically.
    """
    def __init__(self, name, dict_of_expressions, selection=None):
        """Ctor

        :param name: name of the catalog of expressions
        :type name: str

        :param dict_of_expressions: dictionary where the keys are the
            names of each expression, and the values the expressions
            themselves.
        :type dict_of_expressions: dict(str: biogeme.expressions.Expression)

        :param selection: name of the specifciation that is selected
        :type selection: str

        :raise biogemeError: if dict_of_expressions is empty
        """
        super().__init__()

        if not dict_of_expressions:
            raise excep.biogemeError(
                f'{name}: cannot create a catalog from an empty dictionary'
            )
        self.name = name
        self.dict_of_expressions = dict_of_expressions

        # We store the keys in a list in order to guarantee the order
        self.list_of_names = list(self.dict_of_expressions.keys())
        
        if selection is None:
            self.current_index = 0
        else:
            try:
                self.current_index = self.list_of_names.index(selection)
            except ValueError:
                error_msg = (
                    f'Specification {selection} is not in the list: {self.list_of_names}'
                )
                raise excep.biogemeError(error_msg)

    def selected(self):
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: tuple(str, biogeme.expressions.Expression
        """
        name = self.list_of_names[self.current_index]
        return (name, self.dict_of_expressions[name])

    def contains_multiple_expressions(self):
        """ Check if the expression contains multiple expressions

        :return: True if one of the expressions is of type MultipleExpression
        :rtype: bool
        """
        return True
    
