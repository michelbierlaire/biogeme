"""Defines  a catalog of expressions that may be considered in a specification

Michel Bierlaire
Wed Apr 16 18:35:02 2025

"""

from __future__ import annotations

import logging
from typing import Iterator

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    MultipleExpression,
    NamedExpression,
    validate_and_convert,
)

from .controller import Controller

logger = logging.getLogger(__name__)


class Catalog(MultipleExpression):
    """Catalog of expressions that are interchangeable. Only one of
    them defines the specification. They are designed to be
    modified algorithmically by a controller.
    """

    def __init__(
        self,
        catalog_name: str,
        named_expressions: list[NamedExpression],
        controlled_by: Controller | None = None,
    ):
        """Ctor

        :param catalog_name: name of the catalog of expressions

        :param named_expressions: list of NamedExpression,
            each containing a name and an expression.

        :param controlled_by: Object controlling the selection of the specifications.

        :raise BiogemeError: if list_of_named_expressions is empty
        :raise BiogemeError: if incompatible Controller

        """
        super().__init__(catalog_name)

        if not named_expressions:
            raise BiogemeError(
                f'{catalog_name}: cannot create a catalog from an empty list.'
            )

        if controlled_by and not isinstance(controlled_by, Controller):
            error_msg = (
                f'The controller must be of type Controller and not '
                f'{type(controlled_by)}'
            )
            raise BiogemeError(error_msg)

        self.named_expressions = [
            NamedExpression(
                name=named.name,
                expression=validate_and_convert(named.expression),
            )
            for named in named_expressions
        ]

        # Declare the expressions as children of the catalog
        for named_expression in self.named_expressions:
            self.children.append(named_expression.expression)

        names = [named_expr.name for named_expr in self.named_expressions]
        if controlled_by is None:
            controller_name = catalog_name
            self.controlled_by = Controller(
                controller_name=controller_name, specification_names=names
            )
        else:
            self.controlled_by = controlled_by
            controller_names = list(controlled_by.specification_names)
            if names != controller_names:
                error_msg = (
                    f'Incompatible IDs between catalog [{names}] and controller '
                    f'[{controller_names}]'
                )
                raise BiogemeError(error_msg)

    def get_all_controllers(self) -> set[Controller]:
        """Provides all controllers controlling the specifications of
            a multiple expression

        :return: a set of controllers
        :rtype: set(biogeme.controller.Controller)

        """
        all_controllers = {self.controlled_by}
        for e in self.children:
            all_controllers |= e.get_all_controllers()
        return all_controllers

    @classmethod
    def from_dict(
        cls,
        catalog_name: str,
        dict_of_expressions: dict[str, Expression],
        controlled_by: Controller | None = None,
    ) -> Catalog:
        """Ctor using a dict instead of a list.

        Python used not to guarantee the order of elements of a dict,
        although, in practice, it is always preserved. If the order is
        critical, it is better to use the main constructor. If not,
        this constructor provides a more readable code.

        :param catalog_name: name of the catalog
        :param dict_of_expressions: dict associating the name of an
            expression and the expression itself.
        :param controlled_by: Object controlling the selection of the specifications.

        """
        named_expressions = [
            NamedExpression(name=name, expression=expression)
            for name, expression in dict_of_expressions.items()
        ]
        return cls(
            catalog_name=catalog_name,
            named_expressions=named_expressions,
            controlled_by=controlled_by,
        )

    def __iter__(self) -> Iterator[NamedExpression]:
        """Obtain an iterator on the named expressions"""
        return self.get_iterator()

    def get_iterator(self) -> Iterator[NamedExpression]:
        """Obtain an iterator on the named expressions"""
        return iter(self.named_expressions)

    def selected(self) -> NamedExpression:
        """Return the selected expression and its name

        :return: the name and the selected expression
        :rtype: NamedExpression
        """
        return self.named_expressions[self.controlled_by.current_index]
