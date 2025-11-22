"""Arithmetic expressions accepted by Biogeme: elementary expressions

Michel Bierlaire
Tue Mar 25 17:34:47 2025
"""

from __future__ import annotations

import logging

from .base_expressions import Expression
from .elementary_types import TypeOfElementaryExpression

logger = logging.getLogger(__name__)


class Elementary(Expression):
    """Elementary expression.

    It is typically defined by a name appearing in an expression. It
    can be a variable (from the database), or a parameter (fixed or to
    be estimated using maximum likelihood), a random variable for
    numerical integration, or Monte-Carlo integration.

    """

    expression_type = None

    def __init__(self, name: str):
        """Constructor

        :param name: name of the elementary expression.
        :type name: string

        """
        super().__init__()
        self.name = name  #: name of the elementary expression

        # self.elementary_index = None
        """The index should be unique for all elementary expressions
        appearing in a given set of formulas.
        """
        self.specific_id: int | None = None  # Index of the element in its own array.

    def __str__(self) -> str:
        """string method

        :return: name of the expression
        :rtype: str
        """
        return f"{self.name}"

    def __repr__(self):
        return f'<{self.get_class_name()} name={self.name}>'

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
        self, old_name: str, new_name: str, elementary_type: TypeOfElementaryExpression
    ) -> int:
        """Rename an elementary expression
        :return: number of modifications actually performed
        """
        if self.expression_type == elementary_type and self.name == old_name:
            self.name = new_name
            return 1
        return 0

    def set_specific_id(self, name, specific_id, the_type: TypeOfElementaryExpression):
        """The elementary IDs identify the position of each element in the corresponding database"""
        if the_type == self.expression_type and name == self.name:
            self.specific_id = specific_id
