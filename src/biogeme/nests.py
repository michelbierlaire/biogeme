"""Implements objects in charge of the management of nests for the
nested and the cross-nested logit model.

:author: Michel Bierlaire
:date: Thu Oct  5 14:45:58 2023

"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Iterator, Any

import numpy as np
import pandas as pd
from scipy.integrate import dblquad

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    Numeric,
    ExpressionOrNumeric,
    Beta,
    get_dict_expressions,
    get_dict_values,
)


# The first versions of Biogeme defined the nests as follows. We keep his for backward compatibility.
OldOneNestForNestedLogit = tuple[ExpressionOrNumeric, list[int]]
OldNestsForNestedLogit = tuple[OldOneNestForNestedLogit, ...]
OldOneNestForCrossNestedLogit = tuple[ExpressionOrNumeric, dict[int, Expression]]
OldNestsForCrossNestedLogit = tuple[OldOneNestForCrossNestedLogit, ...]

logger = logging.getLogger(__name__)


@dataclass
class OneNestForNestedLogit:
    """Class capturing the information for one nest of the nested logit model"""

    nest_param: ExpressionOrNumeric
    list_of_alternatives: list[int]
    name: str | None = None

    @classmethod
    def from_tuple(cls, the_tuple: OldOneNestForNestedLogit) -> OneNestForNestedLogit:
        """Ctor to initialize the nest using the old syntax of Biogeme with a tuple"""
        return cls(*the_tuple)

    def intersection(self, other_nest: OneNestForNestedLogit) -> set[int]:
        """Returns the intersection of two nests. Designed to verify the
        validity of the specification

        """
        set1 = set(self.list_of_alternatives)
        set2 = set(other_nest.list_of_alternatives)
        return set1 & set2


def get_nest(a_nest: OneNestForNestedLogit | OldOneNestForNestedLogit):
    """Convert a nest definition to an OneNestForNestedLogot, if needed."""

    if isinstance(a_nest, OneNestForNestedLogit):
        return a_nest
    if isinstance(a_nest, tuple):
        return OneNestForNestedLogit.from_tuple(a_nest)
    raise TypeError(f'Object to type {type(a_nest)} does not represent a nest.')


@dataclass
class OneNestForCrossNestedLogit:
    """Tuple capturing the information for one nest of the cross-nested logit"""

    nest_param: ExpressionOrNumeric
    dict_of_alpha: dict[int, ExpressionOrNumeric]

    name: str | None = None

    @classmethod
    def from_tuple(
        cls, the_tuple: OldOneNestForCrossNestedLogit
    ) -> OneNestForCrossNestedLogit:
        """Ctor to initialize the nest using the old syntax of Biogeme with a tuple"""
        return cls(*the_tuple)

    def __post_init__(self) -> None:
        self.dict_of_alpha = get_dict_expressions(self.dict_of_alpha)
        self.list_of_alternatives = list(self.dict_of_alpha.keys())

    def all_alpha_fixed(self) -> bool:
        """Check if the alpha parameters have a numeric value."""
        for _, alpha in self.dict_of_alpha.items():
            if isinstance(alpha, Beta):
                if alpha.status == 0:
                    return False
            elif isinstance(alpha, Expression) and not isinstance(alpha, Numeric):
                return False
        return True


class Nests:
    """Generic interface for the nests."""

    def __init__(
        self,
        choice_set: list[int],
        tuple_of_nests: tuple[OneNestForNestedLogit | OneNestForCrossNestedLogit, ...],
    ):
        """Ctor

        :param choice_set: the list of all alternatives in the choice
            set. We use a list instead of a set because the order
            matters.

        :param tuple_of_nests: the list of nests

        """

        self.choice_set = choice_set
        self.tuple_of_nests = tuple_of_nests

        # Provide names to nests if there is none.
        nest_counter = 0
        for nest in self.tuple_of_nests:
            nest_counter += 1
            if nest.name is None:
                nest.name = f'nest_{nest_counter}'

        # Set of alternatives involved in nests
        self.mev_alternatives = set().union(
            *(set(nest.list_of_alternatives) for nest in self.tuple_of_nests)
        )
        # Verify that all elements are indeed in the choice set
        invalid_elements = self.mev_alternatives - set(self.choice_set)
        if invalid_elements:
            raise BiogemeError(
                f'The following alternatives appear in a nest and not in the '
                f'choice set: {invalid_elements}'
            )

        self.alone = set(self.choice_set) - self.mev_alternatives
        if self.alone:
            warning_msg = (
                f'The following elements do not appear in any nest and are assumed each to be alone in a separate '
                f'nest: {self.alone}. If it is not the intention, check the assignment of alternatives to nests.'
            )
            logger.warning(warning_msg)

    def __getitem__(
        self, index: int
    ) -> OneNestForNestedLogit | OneNestForCrossNestedLogit:
        if index < 0 or index >= len(self.tuple_of_nests):
            raise IndexError(
                f'Index out of bounds. Valid indices are between 0 and '
                f'{len(self.tuple_of_nests) - 1}.'
            )
        return self.tuple_of_nests[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.tuple_of_nests)

    def check_names(self) -> bool:
        """Checks that all the nests have a name"""
        for nest in self.tuple_of_nests:
            if nest.name is None:
                return False
        return True

    def check_union(self) -> tuple[bool, str]:
        """Check if the union of the nests is the choice set

        :return: a boolean with the result of the check, as a message if check fails.
        """

        # Union Check: The union of all lists should be equal to choice_set

        union_of_lists = {
            i for nest in self.tuple_of_nests for i in nest.list_of_alternatives
        }
        union_of_lists |= set(self.alone)
        if union_of_lists != set(self.choice_set):
            missing_values = set(self.choice_set) - union_of_lists  # set difference
            extra_values = union_of_lists - set(
                self.choice_set
            )  # values not in choice_set
            error_msg_1 = (
                f'Alternatives in the choice set, '
                f'but not in any nest: {missing_values}.'
                if missing_values
                else ''
            )
            error_msg_2 = (
                f'Alternatives in a nest, but not in the choice set: {extra_values}.'
                if extra_values
                else ''
            )
            error_msg = f'{error_msg_1} {error_msg_2}'
            logger.error(error_msg)
            return False, error_msg
        return True, ''


class NestsForNestedLogit(Nests):
    """This class handles nests for the nested logit model"""

    def __init__(
        self,
        choice_set: list[int],
        tuple_of_nests: tuple[OneNestForNestedLogit, ...] | OldNestsForNestedLogit,
    ):
        """Ctor

        :param choice_set: the list of all alternatives in the choice set
        :param tuple_of_nests: the list of nests. The old syntax can still be used:
            A tuple containing as many items as nests.
            Each item is also a tuple containing two items:

            - an object of type biogeme.expressions.expr.Expression representing
              the nest parameter,
            - a list containing the list of identifiers of the alternatives
              belonging to the nest.

            Example::

                nesta = MUA ,[1, 2, 3]
                nestb = MUB ,[4, 5, 6]
                nests = nesta, nestb


        """

        # In previous versions of Biogeme, the nests were defined
        # using regular tuples, not NamedTuple. We cast them here for
        # the sake of backward compatibility.
        if not all(isinstance(elem, OneNestForNestedLogit) for elem in tuple_of_nests):
            tuple_of_nests = tuple(
                OneNestForNestedLogit.from_tuple(nest) for nest in tuple_of_nests
            )
        super().__init__(choice_set, tuple_of_nests)

    def correlation(
        self,
        parameters: dict[str, float] | None = None,
        alternatives_names: dict[int, str] | None = None,
        mu: float = 1.0,
    ) -> pd.DataFrame:
        """Calculate the correlation matrix of the error terms of all
            alternatives of a nested logit model.

        :param parameters: values of the parameters.

        :param alternatives_names: dictionary associating a name with each alternative.

        :param mu: value of the scale parameter mu.

        :return: correlation matrix
        """
        index = {alt: i for i, alt in enumerate(self.choice_set)}
        nbr_of_alternatives = len(self.choice_set)
        if alternatives_names is None:
            alternatives_names = {i: str(i) for i in self.choice_set}
        correlation = np.identity(nbr_of_alternatives)
        for m in self.tuple_of_nests:
            if isinstance(m.nest_param, Expression):
                if parameters:
                    m.nest_param.change_init_values(parameters)
                mu_m = m.nest_param.get_value()
            else:
                mu_m = m.nest_param
            alt_m = m.list_of_alternatives
            for i, j in itertools.combinations(alt_m, 2):
                correlation[index[i]][index[j]] = correlation[index[j]][index[i]] = (
                    1.0 - 1.0 / (mu_m * mu_m)
                    if mu == 1.0
                    else 1.0 - (mu * mu) / (mu_m * mu_m)
                )
        return pd.DataFrame(
            correlation,
            index=list(alternatives_names.values()),
            columns=list(alternatives_names.values()),
        )

    def check_intersection(self) -> tuple[bool, str]:
        """Check if the intersection of nests is empty

        :return: a boolean with the result of the check, and a message if check fails.

        """
        for i, nest in enumerate(self.tuple_of_nests):
            if set(nest.list_of_alternatives) & self.alone:
                error_msg = (
                    f'The following alternatives are both in nest {nest.name} '
                    f'and identified as alone in a nest: '
                    f'{set(nest.list_of_alternatives) & self.alone}'
                )
                logger.error(error_msg)
                return False, error_msg
            for j, other_nest in enumerate(self.tuple_of_nests):
                if i != j:
                    the_intersection = nest.intersection(other_nest)
                    if the_intersection:
                        error_msg = (
                            f'The following alternatives appear both in nests '
                            f'{nest.name} and {other_nest.name}: {the_intersection}.'
                        )
                        logger.error(error_msg)
                        return False, error_msg
        return True, ''

    def check_partition(self) -> tuple[bool, str]:
        """Check if the nests correspond to a partition

        :return: a boolean with the result of the check, and a message if check fails.
        """
        valid_union, msg_union = self.check_union()
        valid_intersection, msg_intersection = self.check_intersection()
        return valid_union and valid_intersection, msg_union + '; ' + msg_intersection


class NestsForCrossNestedLogit(Nests):
    """This class handles nests for the cross-nested logit model"""

    def __init__(
        self,
        choice_set: list[int],
        tuple_of_nests: (
            tuple[OneNestForCrossNestedLogit, ...] | OldNestsForCrossNestedLogit
        ),
    ):
        """Ctor

        :param choice_set: the list of all alternatives in the choice set
        :param tuple_of_nests: the list of nests

        """

        # In previous versions of Biogeme, the nests were defined
        # using regular tuples, not NamedTuple. We cast them here for
        # the sake of backward compatibility.
        if not all(
            isinstance(elem, OneNestForCrossNestedLogit) for elem in tuple_of_nests
        ):
            tuple_of_nests = tuple(
                OneNestForCrossNestedLogit.from_tuple(nest) for nest in tuple_of_nests
            )
        super().__init__(choice_set, tuple_of_nests)

    def all_alphas_fixed(self) -> bool:
        """Check if all the alphas are fixed"""
        for nest in self.tuple_of_nests:
            if not nest.all_alpha_fixed():
                return False
        return True

    def get_alpha_dict(self, alternative_id: int) -> dict[str, Expression]:
        """Generates a dict mapping each nest with the alpha
        parameter, for a given alternative

        :param alternative_id: identifier of the alternative
        :return: a dict mapping the name of a nest and the alpha expression
        """
        alphas = {
            nest.name: nest.dict_of_alpha.get(alternative_id, Numeric(0.0))
            for nest in self.tuple_of_nests
        }
        return alphas

    def get_alpha_values(self, alternative_id: int) -> dict[str, float]:
        """Generates a dict mapping each nest with the value of the alpha
        parameters, for a given alternative

        :param alternative_id: identifier of the alternative
        :return: a dict mapping the name of a nest and the value of the alpha expression
        """

        alpha_dict = self.get_alpha_dict(alternative_id=alternative_id)
        alpha_values = {
            key: expression.get_value() for key, expression in alpha_dict.items()
        }
        return alpha_values

    def check_validity(self) -> tuple[bool, str]:
        """Verifies if the cross-nested logit specification is valid

        :return: a boolean with the result of the check, and a message if check fails.
        """
        ok, message = self.check_union()

        alt: dict[int, list[Expression]] = {i: [] for i in self.choice_set}
        number = 0
        for nest in self.tuple_of_nests:
            alpha = nest.dict_of_alpha
            for i, a in alpha.items():
                if a.get_value() != 0.0:
                    alt[i].append(a)
            number += 1

        problems_one = []
        for i, ell in alt.items():
            if len(ell) == 1 and isinstance(ell[0], Expression):
                problems_one.append(i)

        if problems_one:
            message += (
                f' Alternative in exactly one nest, '
                f'and parameter alpha is defined by an '
                f'expression, and may not be constant: {problems_one}'
            )

        return ok, message

    def covariance(
        self, i: int, j: int, parameters: dict[str, float] | None = None
    ) -> float:
        """Calculate the covariance between the error terms of two
        alternatives of a cross-nested logit model. It is assumed that
        the homogeneity parameter mu of the model has been normalized
        to one.

        :param parameters: values of the parameters.

        :param i: first alternative

        :param j: second alternative

        :return: value of the correlation

        :raise BiogemeError: if the requested number is non-positive or a float

        """

        if i not in self.choice_set:
            raise BiogemeError(f'Unknown alternative: {i}')
        if j not in self.choice_set:
            raise BiogemeError(f'Unknown alternative: {j}')

        if i == j:
            return np.pi * np.pi / 6.0

        def integrand(z_i: float, z_j: float) -> float:
            """Function to be integrated to calculate the correlation between
            alternative i and alternative j.

            :param z_i: argument corresponding to alternative i
            :type z_i: float

            :param z_j: argument corresponding to alternative j
            :type z_j: float
            """
            y_i = -np.log(z_i)
            y_j = -np.log(z_j)
            xi_i = -np.log(y_i)
            xi_j = -np.log(y_j)
            dy_i = -1.0 / z_i
            dy_j = -1.0 / z_j
            dxi_i = -dy_i / y_i
            dxi_j = -dy_j / y_j

            g_sum = 0.0
            gi_sum = 0.0
            gj_sum = 0.0
            gij_sum = 0.0
            for m in self.tuple_of_nests:
                if isinstance(m.nest_param, Expression):
                    if parameters:
                        m.nest_param.change_init_values(parameters)
                    mu_m = m.nest_param.get_value()
                else:
                    mu_m = m.nest_param
                alphas = get_dict_values(m.dict_of_alpha, parameters)
                alpha_i = alphas.get(i, 0)
                if alpha_i != 0:
                    term_i = (alpha_i * y_i) ** mu_m
                else:
                    term_i = 0
                alpha_j = alphas.get(j, 0)
                if alpha_j != 0:
                    term_j = (alpha_j * y_j) ** mu_m
                else:
                    term_j = 0
                the_sum = term_i + term_j
                p1 = (1.0 / mu_m) - 1
                p2 = (1.0 / mu_m) - 2
                g_sum += the_sum ** (1.0 / mu_m)
                if alpha_i != 0:
                    gi_sum += alpha_i**mu_m * y_i ** (mu_m - 1) * the_sum**p1
                if alpha_j != 0:
                    gj_sum += alpha_j**mu_m * y_j ** (mu_m - 1) * the_sum**p1
                if mu_m != 1.0 and alpha_i != 0 and alpha_j != 0:
                    gij_sum += (
                        (1 - mu_m)
                        * the_sum**p2
                        * (alpha_i * alpha_j) ** mu_m
                        * (y_i * y_j) ** (mu_m - 1)
                    )

            f = np.exp(-g_sum)
            f_second = f * y_i * y_j * (gi_sum * gj_sum - gij_sum)

            return xi_i * xi_j * f_second * dxi_i * dxi_j

        integral, _ = dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)
        return integral - np.euler_gamma * np.euler_gamma

    def correlation(
        self,
        parameters: dict[str, float] | None = None,
        alternatives_names: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """Calculate the correlation matrix of the error terms of all
            alternatives of cross-nested logit model.

        :param parameters: values of the parameters.
        :param alternatives_names: names of the alternative, for
            better reporting. If not provided, the number are used.
        :return: correlation matrix

        """
        nbr_of_alternatives = len(self.choice_set)
        if alternatives_names is None:
            alternatives_names = {i: str(i) for i in self.choice_set}

        covar = np.empty((nbr_of_alternatives, nbr_of_alternatives))
        for i, alt_i in enumerate(self.choice_set):
            for j, alt_j in enumerate(self.choice_set):
                covar[i][j] = self.covariance(alt_i, alt_j, parameters)
                if i != j:
                    covar[j][i] = covar[i][j]

        v = np.sqrt(np.diag(covar))
        outer_v = np.outer(v, v)
        correlation = covar / outer_v
        correlation[covar == 0] = 0
        return pd.DataFrame(
            correlation,
            index=list(alternatives_names.values()),
            columns=list(alternatives_names.values()),
        )
