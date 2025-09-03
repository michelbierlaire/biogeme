"""Combine several arithmetic expressions and a database to obtain formulas

:author: Michel Bierlaire
:date: Sat Jul 30 12:36:40 2022
"""

from __future__ import annotations

import logging
from typing import Generic, Iterable, NamedTuple, TYPE_CHECKING, TypeVar

import numpy as np

from biogeme.exceptions import BiogemeError

if TYPE_CHECKING:
    from biogeme.database import Database
from biogeme.expressions import (
    Expression,
    TypeOfElementaryExpression,
    Beta,
    Variable,
    RandomVariable,
    Draws,
    list_of_fixed_betas_in_expression,
    list_of_free_betas_in_expression,
    list_of_random_variables_in_expression,
    list_of_draws_in_expression,
)

T = TypeVar('T', bound='Elementary')

try:

    class ElementsTuple(Generic[T], NamedTuple):
        """Data structure for elementary expressions."""

        expressions: dict[str, T] | None
        indices: dict[str, int] | None
        names: list[str]

except TypeError:
    # This exception is raised by Python 3.10.
    class _ElementsTuple(NamedTuple):
        """Data structure for elementary expressions."""

        expressions: dict[str, T] | None
        indices: dict[str, int] | None
        names: list[str]

    class ElementsTuple(Generic[T], _ElementsTuple):
        pass


logger = logging.getLogger(__name__)


def expressions_names_indices(dict_of_elements: dict[str, type[T]]) -> ElementsTuple[T]:
    """Assigns consecutive indices to expressions

    :param dict_of_elements: dictionary of expressions. The keys
        are the names.

    :return: a tuple with the original dictionary, the indices,
        and the sorted names.
    :rtype: ElementsTuple
    """

    indices = {}
    names = sorted(dict_of_elements)

    for i, v in enumerate(names):
        indices[v] = i

    return ElementsTuple(expressions=dict_of_elements, indices=indices, names=names)


class ExpressionRegistry:
    """Class combining managing the ids of an arithmetic expression."""

    def __init__(
        self,
        expressions: Iterable[Expression],
        database: Database,
    ):
        """Ctor

        :param expressions: list of expressions
        :type expressions: list(biogeme.expressions.Expression)

        :param database: database with the variables as column names
        :type database: biogeme.database.Database

        :raises BiogemeError: if an expression contains a variable and
            no database is provided.

        """
        self._expressions: list[Expression] = list(expressions)
        self.database: Database = database
        self._free_betas: list[Beta] | None = None
        self._fixed_betas: list[Beta] | None = None
        self._random_variables: list[RandomVariable] | None = None
        self._draws: list[Draws] | None = None
        self._variables: list[Variable] | None = None
        self._bounds: list[tuple[float, float]] | None = None
        self._require_draws: bool | None = None
        self._free_betas_indices: dict[str, int] | None = None
        self._fixed_betas_indices: dict[str, int] | None = None
        self._variables_indices: dict[str, int] | None = None
        self._random_variables_indices: dict[str, int] | None = None
        self._draws_indices: dict[str, int] | None = None
        self.broadcast()

    @property
    def bounds(self) -> list[tuple[float, float]]:
        if self._bounds is None:
            self._bounds = [
                (beta.lower_bound, beta.upper_bound) for beta in self._free_betas
            ]
        return self._bounds

    @property
    def expressions(self) -> list[Expression]:
        return self._expressions

    @property
    def requires_draws(self) -> bool:
        if self._require_draws is None:
            self._require_draws = any(
                expression.requires_draws() for expression in self.expressions
            )
        return self._require_draws

    @property
    def free_betas_names(self) -> list[str]:
        return [beta.name for beta in self.free_betas]

    @property
    def draws_names(self) -> list[str]:
        return [draw.name for draw in self.draws]

    @property
    def free_betas(self) -> list[Beta]:
        if self._free_betas is None:
            self._free_betas = []
            for f in self.expressions:
                self._free_betas.extend(list_of_free_betas_in_expression(f))
            # Remove duplicates based on the name attribute
            unique_betas = {beta.name: beta for beta in self._free_betas}
            self._free_betas = list(unique_betas.values())
        return self._free_betas

    @property
    def free_betas_indices(self) -> dict[str, int]:
        if self._free_betas_indices is None:
            self._free_betas_indices = {
                beta.name: i for i, beta in enumerate(self.free_betas)
            }
        return self._free_betas_indices

    @property
    def fixed_betas(self) -> list[Beta]:
        if self._fixed_betas is None:
            self._fixed_betas = []
            for f in self.expressions:
                self._fixed_betas.extend(list_of_fixed_betas_in_expression(f))
            # Remove duplicates based on the name attribute
            unique_betas = {beta.name: beta for beta in self._fixed_betas}
            self._fixed_betas = list(unique_betas.values())
        return self._fixed_betas

    @property
    def fixed_betas_indices(self) -> dict[str, int]:
        if self._fixed_betas_indices is None:
            self._fixed_betas_indices = {
                beta.name: i for i, beta in enumerate(self.fixed_betas)
            }
        return self._fixed_betas_indices

    @property
    def random_variables(self) -> list[RandomVariable]:
        if self._random_variables is None:
            self._random_variables = []
            for f in self.expressions:
                self._random_variables.extend(list_of_random_variables_in_expression(f))
            # Remove duplicates based on the name attribute
            unique_rv = {rv.name: rv for rv in self._random_variables}
            self._random_variables = list(unique_rv.values())
        return self._random_variables

    @property
    def random_variables_indices(self) -> dict[str, int]:
        if self._random_variables_indices is None:
            self._random_variables_indices = {
                rv.name: i for i, rv in enumerate(self.random_variables)
            }
        return self._random_variables_indices

    @property
    def draws(self) -> list[Draws]:
        if self._draws is None:
            self._draws = []
            for f in self.expressions:
                self._draws.extend(list_of_draws_in_expression(f))
            # Remove duplicates based on the name attribute
            unique_draws = {draw.name: draw for draw in self._draws}
            self._draws = list(unique_draws.values())
        return self._draws

    @property
    def draws_indices(self) -> dict[str, int]:
        if self._draws_indices is None:
            self._draws_indices = {draw.name: i for i, draw in enumerate(self.draws)}
        return self._draws_indices

    @property
    def variables(self) -> list[Variable]:
        if self._variables is None:
            self._variables = (
                []
                if self.database is None
                else [
                    Variable(name) for name in self.database.dataframe.columns.to_list()
                ]
            )
        return self._variables

    @property
    def variables_indices(self) -> dict[str, int]:
        if self._variables_indices is None:
            self._variables_indices = {
                variable.name: i for i, variable in enumerate(self.variables)
            }
        return self._variables_indices

    @property
    def number_of_free_betas(self) -> int:
        return len(self.free_betas)

    @property
    def number_of_random_variables(self) -> int:
        return len(self.random_variables)

    @property
    def free_betas_init_values(self) -> dict[str, float]:
        """
        Retrieve the initial values of all free beta parameters.

        :return: A dictionary mapping parameter names to their initial values.
        """
        return {
            expression.name: float(expression.init_value)
            for expression in self.free_betas
        }

    def get_betas_array(self, betas_dict: dict[str, float]) -> np.ndarray:
        try:
            result = np.array(
                [float(betas_dict[beta.name]) for beta in self.free_betas]
            )
            return result
        except KeyError:
            betas_input = set(betas_dict.keys())
            betas_names = set([beta.name for beta in self.free_betas])
            unknown_betas = betas_input - betas_names
            err_msg = f'Unknown parameters: {unknown_betas}. List of known parameters: {betas_names}'
            raise BiogemeError(err_msg)

    def get_complete_betas_array(self, betas_dict: dict[str, float]) -> np.ndarray:
        complete_dict = self.complete_dict_of_free_beta_values(the_betas=betas_dict)
        return np.array([float(complete_dict[beta.name]) for beta in self.free_betas])

    def get_named_betas_values(self, values: np.ndarray) -> dict[str, float]:
        return {beta.name: float(values[i]) for i, beta in enumerate(self.free_betas)}

    @property
    def list_of_free_betas_init_values(self) -> list[float]:
        """
        Retrieve the initial values of all free beta parameters as a list.

        :return: A list with parameter initial values.
        """
        return list(self.free_betas_init_values.values())

    def complete_dict_of_free_beta_values(
        self, the_betas: dict[str, float]
    ) -> dict[str, float]:
        """Build a list of values for the beta parameters, some of them provided by the user. The others are the
        initial values.

        :param the_betas: user provided values
        :return: A list with parameter  values.
        """
        for beta in self.free_betas:
            if beta.name not in the_betas:
                the_betas[beta.name] = beta.init_value
        return the_betas

    @property
    def fixed_betas_values(self) -> dict[str, float]:
        """
        Retrieve the initial values of all fixed beta parameters.

        :return: A dictionary mapping parameter names to their fixed values.
        """
        return {
            expression.name: expression.init_value for expression in self.fixed_betas
        }

    def draw_types(self) -> dict[str, str]:
        """Retrieve the type of draw for each draw expression"""
        return {expression.name: expression.draw_type for expression in self.draws}

    def broadcast(self) -> None:
        """Broadcast the ids to the expressions"""
        for expression in self.expressions:
            for name, index in self.free_betas_indices.items():
                expression.set_specific_id(
                    name, index, TypeOfElementaryExpression.FREE_BETA
                )
            for name, index in self.fixed_betas_indices.items():
                expression.set_specific_id(
                    name, index, TypeOfElementaryExpression.FIXED_BETA
                )
            for name, index in self.variables_indices.items():
                expression.set_specific_id(
                    name, index, TypeOfElementaryExpression.VARIABLE
                )
            for name, index in self.draws_indices.items():
                expression.set_specific_id(
                    name, index, TypeOfElementaryExpression.DRAWS
                )
            for name, index in self.random_variables_indices.items():
                expression.set_specific_id(
                    name, index, TypeOfElementaryExpression.RANDOM_VARIABLE
                )
