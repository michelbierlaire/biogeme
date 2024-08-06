"""Combine several arithmetic expressions and a database to obtain formulas

:author: Michel Bierlaire
:date: Sat Jul 30 12:36:40 2022
"""

from __future__ import annotations

import logging
from typing import (
    NamedTuple,
    TYPE_CHECKING,
    TypeVar,
    Generic,
    Iterable,
)

import numpy as np
import pandas as pd

from biogeme.exceptions import BiogemeError
from ..deprecated import deprecated

if TYPE_CHECKING:
    from .base_expressions import Expression, Elementary
    from . import Beta, RandomVariable, bioDraws, Variable
    from ..database import Database

from .elementary_types import TypeOfElementaryExpression

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


class IdManager:
    """Class combining managing the ids of an arithmetic expression."""

    def __init__(
        self,
        expressions: Iterable[Expression],
        database: Database,
        number_of_draws: int,
    ):
        """Ctor

        :param expressions: list of expressions
        :type expressions: list(biogeme.expressions.Expression)

        :param database: database with the variables as column names
        :type database: biogeme.database.Database

        :param number_of_draws: number of draws for Monte-Carlo integration
        :type number_of_draws: int

        :raises BiogemeError: if an expression contains a variable and
            no database is provided.

        """
        self.expressions: list[Expression] = list(expressions)
        self.database: Database = database
        self.number_of_draws: int = number_of_draws
        self.elementary_expressions: ElementsTuple[Elementary] | None = None
        self.free_betas: ElementsTuple[Beta] | None = None
        self.free_betas_values: np.ndarray | None = None
        self.number_of_free_betas: int = 0
        self.fixed_betas: ElementsTuple[Beta] | None = None
        self.fixed_betas_values: np.ndarray | None = None
        self.bounds: list[tuple[float, float]] | None = None
        self.random_variables: ElementsTuple[RandomVariable] | None = None
        self.draws: ElementsTuple[bioDraws] | None = None
        self.variables: ElementsTuple[Variable] | None = None
        self.requires_draws: bool = False
        for f in self.expressions:
            the_variables = f.set_of_elementary_expression(
                the_type=TypeOfElementaryExpression.VARIABLE
            )
            if the_variables and database is None:
                raise BiogemeError(
                    f'No database is provided and an expression '
                    f'contains variables: {the_variables}'
                )
            if f.embed_expression('MonteCarlo') or f.embed_expression('bioDraws'):
                self.requires_draws = True

        self.prepare()

    def __str__(self) -> str:
        return str(self.elementary_expressions.indices)

    def __repr__(self) -> str:
        return str(self.elementary_expressions.indices)

    def __eq__(self, other) -> bool:
        return self.elementary_expressions == other.elementary_expressions

    def draw_types(self) -> dict[str, str]:
        """Retrieve the type of draw for each draw expression"""
        return {
            name: expression.drawType
            for name, expression in self.draws.expressions.items()
        }

    def audit(self) -> tuple[list[str], list[str]]:
        """Performs various checks on the expressions.

        :return: tuple listOfErrors, listOfWarnings
        :rtype: list(string), list(string)
        """
        list_of_errors = []
        list_of_warnings = []
        if self.database.is_panel():
            for the_expression in self.expressions:
                dict_of_variables = the_expression.check_panel_trajectory()
                if dict_of_variables:
                    err_msg = (
                        f'Error in the loglikelihood function. '
                        f'Some variables are not inside '
                        f'PanelLikelihoodTrajectory: '
                        f'{dict_of_variables} .'
                        f'If the database is organized as panel data, '
                        f'all variables must be used inside a '
                        f'PanelLikelihoodTrajectory. '
                        f'If it is not consistent with your model, '
                        f'generate a flat '
                        f'version of the data using the function '
                        f'`generateFlatPanelDataframe`.'
                    )
                    list_of_errors.append(err_msg)
        return list_of_errors, list_of_warnings

    def prepare(self) -> None:
        """Extract from the formulas the literals (parameters,
        variables, random variables) and decide a numbering convention.

        The numbering is done in the following order:

        (i) free betas,
        (ii) fixed betas,
        (iii) random variables for numerical integration,
        (iv) random variables for Monte-Carlo integration,
        (v) variables

        The numbering convention will be performed for all expressions
        together, so that the same elementary expressions in several
        expressions will have the same index.


        """

        # Free parameters (to be estimated), sorted by alphabetical order
        expr = {}
        for f in self.expressions:
            d = f.dict_of_elementary_expression(
                the_type=TypeOfElementaryExpression.FREE_BETA
            )
            expr = dict(expr, **d)

        self.free_betas = expressions_names_indices(expr)

        self.bounds = [
            (
                self.free_betas.expressions[b].lb,
                self.free_betas.expressions[b].ub,
            )
            for b in self.free_betas.names
        ]
        self.number_of_free_betas = len(self.free_betas.names)
        # Fixed parameters (not to be estimated), sorted by alphabetical order.
        expr = {}
        for f in self.expressions:
            d = f.dict_of_elementary_expression(
                the_type=TypeOfElementaryExpression.FIXED_BETA
            )
            expr = dict(expr, **d)
        self.fixed_betas = expressions_names_indices(expr)

        # Random variables for numerical integration
        expr = {}
        for f in self.expressions:
            d = f.dict_of_elementary_expression(
                the_type=TypeOfElementaryExpression.RANDOM_VARIABLE
            )
            expr = dict(expr, **d)
        self.random_variables = expressions_names_indices(expr)

        # Draws
        expr = {}
        for f in self.expressions:
            d = f.dict_of_elementary_expression(
                the_type=TypeOfElementaryExpression.DRAWS
            )
            expr = dict(expr, **d)
        self.draws = expressions_names_indices(expr)

        # Variables
        # Here, we do not extract the variables from the
        # formulas. Instead, we use all the variables in the database.
        if self.database is not None:
            variables_names = self.database.data.columns.to_list()
            variables_indices = {}
            for i, v in enumerate(variables_names):
                variables_indices[v] = i
            self.variables = ElementsTuple(
                expressions=None,
                indices=variables_indices,
                names=variables_names,
            )
        else:
            self.variables = ElementsTuple(expressions=None, indices=None, names=[])

        # Merge all the names
        elementary_expressions_names = (
            self.free_betas.names
            + self.fixed_betas.names
            + self.random_variables.names
            + self.draws.names
            + self.variables.names
        )

        if len(elementary_expressions_names) != len(set(elementary_expressions_names)):
            duplicates = {
                x
                for x in elementary_expressions_names
                if elementary_expressions_names.count(x) > 1
            }
            error_msg = (
                f'The following elementary expressions are defined '
                f'more than once: {duplicates}.'
            )
            raise BiogemeError(error_msg)

        elementary_expressions_indices = {
            v: i for i, v in enumerate(elementary_expressions_names)
        }

        self.elementary_expressions = ElementsTuple(
            expressions=None,
            indices=elementary_expressions_indices,
            names=elementary_expressions_names,
        )

        self.free_betas_values = [
            self.free_betas.expressions[x].initValue for x in self.free_betas.names
        ]
        self.fixed_betas_values = [
            self.fixed_betas.expressions[x].initValue for x in self.fixed_betas.names
        ]

        if self.requires_draws:
            self.database.generate_draws(
                self.draw_types(), self.draws.names, self.number_of_draws
            )

    def set_data_map(self, sample: pd.DataFrame):
        """Specify the map of the panel data in the expressions

        :param sample: map of the panel data (see
            :func:`biogeme.database.Database.buildPanelMap`)
        :type sample: pandas.DataFrame
        """
        for f in self.expressions:
            f.cpp.set_data_map(sample)

    @deprecated(new_func=set_data_map)
    def setDataMap(self, sample: pd.DataFrame):
        pass

    def set_data(self, sample: pd.DataFrame):
        """Specify the sample

        :param sample: map of the panel data (see
            :func:`biogeme.database.Database.buildPanelMap`)
        :type sample: pandas.DataFrame

        """
        for f in self.expressions:
            f.cpp.set_data(sample)

    @deprecated(new_func=set_data)
    def setData(self, sample: pd.DataFrame):
        pass
