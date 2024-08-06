""" Iterator on arithmetic expressions in a catalog

:author: Michel Bierlaire
:date: Sat Sep  9 16:18:02 2023
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from biogeme.configuration import Configuration

if TYPE_CHECKING:
    from biogeme.expressions import Expression


class SelectedExpressionsIterator:
    """A multiple expression is an expression that contains
    Catalog. This iterator loops on pre-specified configurations
    """

    def __init__(
        self,
        the_expression: Expression,
        configurations: set[biogeme.configuration.Configuration],
    ):
        """Ctor.

        :param the_expression: expression containing Catalogs
        :param configurations: set of configurations
        """
        self.the_expression = the_expression
        self.configurations = configurations
        self.set_iterator = iter(configurations)
        current_configuration = next(self.set_iterator)
        self.the_expression.configure_catalogs(current_configuration)
        self.first = True
        self.number = 0

    def __iter__(self) -> SelectedExpressionsIterator:
        return self

    def __next__(self) -> Expression:
        self.number += 1
        if self.first:
            self.first = False
            return self.the_expression

        current_configuration = next(self.set_iterator)
        self.the_expression.configure_catalogs(current_configuration)
        return self.the_expression
