"""Implements objects in charge of the management of nests for the
nested and the cross-nested logit model.

:author: Michel Bierlaire
:date: Thu Oct  5 14:45:58 2023

"""
import logging
import itertools
from dataclasses import dataclass
from typing import Union, Iterator, Any, Optional
import numpy as np
import pandas as pd
from scipy.integrate import dblquad
from biogeme.expressions import Expression, Numeric
from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)


@dataclass
class OneNestForNestedLogit:
    """Class capturing the information for one nest of the nested logit model"""

    nest_param: Union[Expression, float]
    list_of_alternatives: list[int]
    name: Optional[str] = None

    @classmethod
    def from_tuple(
        cls, the_tuple: tuple[Union[Expression, float], list[int]]
    ) -> 'OneNestForNestedLogit':
        """Ctor to initialize the nest using the old syntax of Biogeme with a tuple"""
        return cls(*the_tuple)

    def intersection(self, other_nest: 'OneNestForNestedLogit') -> set[int]:
        """Returns the intersection of two nests. Designed to verify the
        validity of the specification

        """
        set1 = set(self.list_of_alternatives)
        set2 = set(other_nest.list_of_alternatives)
        return set1 & set2


def get_alpha_expressions(
    the_dict: dict[int, Union[Expression, float]]
) -> dict[int, Expression]:
    """If the dictionary contains float, they are transformed into a
    numerical expression."""

    def generate_expression(value: Union[Expression, float]) -> Expression:
        """Remove the type ambiguity and generates an Expression"""
        if isinstance(value, (int, float)):
            return Numeric(value)
        return value

    return {key: generate_expression(value) for key, value in the_dict.items()}


def get_alpha_values(
    the_dict: dict[int, Union[Expression, float]],
    parameters: Optional[dict[str, float]] = None,
) -> dict[int, float]:
    """If the dictionary contains float, they are transformed into a
    numerical expression.

    :param the_dict: dict of alphas.
    :param parameters: value of the parameters.
    """

    def generate_value(
        value: Union[Expression, float], parameters: dict[str, float]
    ) -> float:
        """Remove the type ambiguity and generates a value"""
        if isinstance(value, (int, float)):
            return value
        if parameters is not None:
            value.set_estimated_values(parameters)
        v = value.getValue()
        return v

    return {key: generate_value(value, parameters) for key, value in the_dict.items()}


@dataclass
class OneNestForCrossNestedLogit:
    """Tuple capturing the information for one nest of the cross-nested logit"""

    nest_param: Union[Expression, float]
    dict_of_alpha: dict[int, Expression]
    name: Optional[str] = None

    @classmethod
    def from_tuple(
        cls, the_tuple: tuple[Union[Expression, float], dict[int, Expression]]
    ) -> 'OneNestForCrossNestedLogit':
        """Ctor to initialize the nest using the old syntax of Biogeme with a tuple"""
        return cls(*the_tuple)

    def __post_init__(self):
        self.dict_of_alpha = get_alpha_expressions(self.dict_of_alpha)
        self.list_of_alternatives = list(self.dict_of_alpha.keys())

    def is_alpha_fixed(self) -> bool:
        """Check if the alpha parameters have a numeric value."""
        for _, alpha in self.dict_of_alpha.items():
            try:
                _ = alpha.getValue()
            except BiogemeError:
                return False
            except AttributeError:
                return True
        return True


class Nests:
    """Generic interface for the nests."""

    def __init__(
        self,
        choice_set: list[int],
        tuple_of_nests: tuple[Any, ...],
    ):
        """Ctor

        :param choice_set: the list of all alternatives in the choice
            set. We use a list instread of a set because thre order
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

    def __getitem__(
        self, index: int
    ) -> Union[OneNestForNestedLogit, OneNestForCrossNestedLogit]:
        if index < 0 or index >= len(self.tuple_of_nests):
            raise IndexError(
                f'Index out of bounds. Valid indices are between 0 and '
                f'{len(self.tuple_of_nests) - 1}.'
            )
        return self.tuple_of_nests[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.tuple_of_nests)

    def check_names(self):
        """Checks that all the nests have a name"""
        for nest in self.tuple_of_nests:
            if nest.name is None:
                return False
        return True

    def check_union(self) -> tuple[bool, str]:
        """Check if the union of the nests is the choice set

        :return: a boolean with the result of the check, as a message is check fails.
        """

        # Union Check: The union of all lists should be equal to choice_set

        union_of_lists = set(
            i for nest in self.tuple_of_nests for i in nest.list_of_alternatives
        )
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
        tuple_of_nests: Union[
            tuple[OneNestForNestedLogit, ...], tuple[tuple[Expression, list[int]], ...]
        ],
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
        :param alone: the list of alternatives alone in one nest. If
            None, any alternative not present in a nest is supposed to
            be alone

        """

        # In previous versions of Biogeme, the nests were defined
        # using regular tuples, not NamedTuple. We cast them here for
        # the sake of backward compatibility.
        def cast_nested_logit(
            one_nest: Union[OneNestForNestedLogit, tuple[Expression, list[int]]]
        ) -> OneNestForNestedLogit:
            if isinstance(one_nest, OneNestForNestedLogit):
                return one_nest
            return OneNestForNestedLogit.from_tuple(one_nest)

        tuple_of_nests = tuple(cast_nested_logit(nest) for nest in tuple_of_nests)

        super().__init__(choice_set, tuple_of_nests)

    def correlation(
        self,
        parameters: dict[str, float] = dict(),
        alternatives_names: Optional[dict[int, str]] = None,
        mu: float = 1.0,
    ) -> 'pd.DataFrame':
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
                m.nest_param.set_estimated_values(parameters)
                mu_m = m.nest_param.getValue_c(prepareIds=True)
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
        tuple_of_nests: tuple[OneNestForCrossNestedLogit, ...],
    ):
        """Ctor

        :param choice_set: the list of all alternatives in the choice set
        :param tuple_of_nests: the list of nests

        """

        # In previous versions of Biogeme, the nests were defined
        # using regular tuples, not NamedTuple. We cast them here for
        # the sake of backward compatibility.
        def cast_cross_nested_logit(one_nest):
            if isinstance(one_nest, OneNestForCrossNestedLogit):
                return one_nest
            return OneNestForCrossNestedLogit.from_tuple(one_nest)

        tuple_of_nests = tuple(cast_cross_nested_logit(nest) for nest in tuple_of_nests)
        super().__init__(choice_set, tuple_of_nests)

    def all_alphas_fixed(self) -> bool:
        """Check if all the alphas are fixed"""
        for nest in self.tuple_of_nests:
            if not nest.is_alpha_fixed():
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
        """Generates a dict mapping each nest with the alpha
        values, for a given alternative

        :param alternative_id: identifier of the alternative
        :return: a dict mapping the name of a nest and the alpha values

        :raise: BiogemeError is one alpha is a non numeric expression
        """

        def get_value(alpha: Union[Expression, float, int]) -> float:
            """Returns a float, irrespectively of the original type

            :param alpha: alpha parameter

            :raise BiogemeError: if the alpha does not have one of the expected types.
            """
            if isinstance(alpha, (int, float)):
                return float(alpha)
            if isinstance(alpha, Expression):
                return alpha.getValue()
            error_msg = f'Unknown type {type(alpha)} for alpha parameter'
            raise BiogemeError(error_msg)

        alphas_expressions = self.get_alpha_dict(alternative_id)
        alphas = {key: get_value(alpha) for key, alpha in alphas_expressions.items()}
        return alphas

    def check_validity(self) -> tuple[bool, str]:
        """Verifies if the cross-nested logit specifciation is valid

        :return: a boolean with the result of the check, and a message if check fails.
        """
        ok, message = self.check_union()

        alt: dict[int, list[Expression]] = {i: [] for i in self.choice_set}
        number = 0
        for nest in self.tuple_of_nests:
            alpha = nest.dict_of_alpha
            for i, a in alpha.items():
                if a != 0.0:
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
        self, i: int, j: int, parameters: dict[str, float] = dict()
    ) -> float:
        """Calculate the covariance between the error terms of two
        alternatives of a cross-nested logit model. It is assumed that
        the homogeneity parameter mu of the model has been normalized
        to one.

        :param parameters: values of the parameters.

        :param i: first alternative

        :param j: second alternative

        :return: value of the correlation

        :raise BiogemeError: if the requested number is non positive or a float

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

            G_sum = 0.0
            Gi_sum = 0.0
            Gj_sum = 0.0
            Gij_sum = 0.0
            for m in self.tuple_of_nests:
                if isinstance(m.nest_param, Expression):
                    m.nest_param.set_estimated_values(parameters)
                    mu_m = m.nest_param.getValue_c(prepareIds=True)
                else:
                    mu_m = m.nest_param
                alphas = get_alpha_values(m.dict_of_alpha, parameters)
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
                G_sum += the_sum ** (1.0 / mu_m)
                if alpha_i != 0:
                    Gi_sum += alpha_i**mu_m * y_i ** (mu_m - 1) * the_sum**p1
                if alpha_j != 0:
                    Gj_sum += alpha_j**mu_m * y_j ** (mu_m - 1) * the_sum**p1
                if mu_m != 1.0 and alpha_i != 0 and alpha_j != 0:
                    Gij_sum += (
                        (1 - mu_m)
                        * the_sum**p2
                        * (alpha_i * alpha_j) ** mu_m
                        * (y_i * y_j) ** (mu_m - 1)
                    )

            F = np.exp(-G_sum)
            F_second = F * y_i * y_j * (Gi_sum * Gj_sum - Gij_sum)

            return xi_i * xi_j * F_second * dxi_i * dxi_j

        integral, _ = dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)
        return integral - np.euler_gamma * np.euler_gamma

    def correlation(
        self,
        parameters: dict[str, float] = dict(),
        alternatives_names: Optional[dict[int, str]] = None,
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
