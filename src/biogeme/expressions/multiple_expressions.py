"""Defines the interface for a catalog of expressions that may be
considered in a specification

Michel Bierlaire
13.04.2025 17:02
"""

import abc
import logging
from typing import Iterator, NamedTuple, final

from biogeme import exceptions
from .base_expressions import Expression

logger = logging.getLogger(__name__)

SEPARATOR = ';'
SELECTION_SEPARATOR = ':'


class NamedExpression(NamedTuple):
    name: str
    expression: Expression


class CatalogItem(NamedTuple):
    catalog_name: str
    item_index: int
    item_name: str


def delegate_to_selected(method_name: str):
    """
    Create a method that delegates its call to `self.selected()[1].<method_name>(...)`.

    This is useful for classes that wrap or dispatch to another `Expression` object,
    avoiding code duplication for common method overrides.

    :param method_name:
        The name of the method to delegate. It must be a valid method name on the
        `Expression` returned by `self.selected()[1]`.
    :return:
        A method that can be assigned to a class and will delegate the call.
    """

    def method(self, *args, **kwargs):
        _, expr = self.selected()
        return getattr(expr, method_name)(*args, **kwargs)

    method.__name__ = method_name
    return method


class MultipleExpression(Expression, metaclass=abc.ABCMeta):
    """Interface for catalog of expressions that are interchangeable. Only one of
    them defines the specification. They are designed to be
    modified algorithmically.
    """

    def __init__(self, the_name: str):
        self.name = the_name  # The name of the expression catalog
        if SEPARATOR in name or SELECTION_SEPARATOR in name:
            error_msg = (
                f'Invalid name: {the_name}. Cannot contain characters '
                f'{SELECTION_SEPARATOR} or {SELECTION_SEPARATOR}'
            )
            raise exceptions.BiogemeError(error_msg)
        super().__init__()

    def deep_flat_copy(self) -> Expression:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` such as this one
        is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        the_expression = self.selected_expression()
        return the_expression.deep_flat_copy()

    @abc.abstractmethod
    def selected(self) -> NamedExpression:
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: tuple(str, biogeme.expressions.Expression)
        """

    @abc.abstractmethod
    def get_iterator(self) -> Iterator[NamedExpression]:
        """Returns an iterator on NamedExpression"""

    @final
    def catalog_size(self) -> int:
        """Provide the size of the catalog

        :return: number of expressions in the catalog
        :rtype: int
        """
        the_iterator = self.get_iterator()

        return len(list(the_iterator))

    @final
    def selected_name(self) -> str:
        """Obtain the name of the selection

        :return: the name of the selected expression
        :rtype: str
        """
        the_name, _ = self.selected()
        return the_name

    @final
    def selected_expression(self) -> NamedExpression:
        """Obtain the selected expression

        :return: the selected expression
        :rtype: biogeme.expressions.Expression
        """
        _, the_expression = self.selected()
        return the_expression

    def __str__(self) -> str:
        named_expression: NamedExpression = self.selected()
        return f'[{self.name}: {named_expression.name}]{named_expression.expression}'


# We now specify all methods that must be delegated to the selected expression
delegated_methods = [
    'set_maximum_number_of_observations_per_individual',
    'change_init_values',
    'rename_elementary',
    'get_elementary_expression',
    'get_value',
    'requires_draws',
    'recursive_construct_jax_function',
    'logit_choice_avail',
    'add_suffix_to_all_variables',
    'set_specific_id',
]

for name in delegated_methods:
    setattr(MultipleExpression, name, final(delegate_to_selected(name)))
