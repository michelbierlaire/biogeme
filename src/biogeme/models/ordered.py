"""Implements various models.

:author: Michel Bierlaire
:date: Fri Mar 29 17:13:14 2019
"""

import logging
from typing import Callable

from biogeme.distributions import logisticcdf
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression, NormalCdf, validate_and_convert

logger = logging.getLogger(__name__)


def build_ordered_thresholds(
    list_of_discrete_values: list[int],
    first_threshold_parameter: Beta,
) -> list[Expression]:
    """Constructs the ordered list of thresholds for the ordered model.

    Given a list of discrete values, constructs the list of thresholds
    [tau_1_2, tau_2_3, ..., tau_{J-1_J}] such that
    tau_1_2 = first_tau, and tau_{j_j+1} = tau_{j-1_j} + diff_j for each intermediate j,
    where diff_j is a Beta parameter.

    :param list_of_discrete_values: list of discrete values (must be at least length 2)
    :param first_threshold_parameter: Beta parameter for the first threshold
    :return: list of Expression thresholds (length len(list_of_discrete_values) - 1)
    """
    if len(list_of_discrete_values) < 2:
        raise BiogemeError('Need at least two discrete values for ordered model.')
    if not isinstance(first_threshold_parameter, Beta):
        raise BiogemeError(
            f'first_tau must be a Beta expression, and not a {type(first_threshold_parameter)}.'
        )
    thresholds = [first_threshold_parameter]
    tau = first_threshold_parameter
    for val in list_of_discrete_values[1:-1]:
        diff = Beta(
            f'{first_threshold_parameter.name}_diff_{val}',
            1,
            0,
            None,
            0,
        )
        tau = tau + diff
        thresholds.append(tau)
    return thresholds


def ordered_likelihood_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
    cdf: Callable[[Expression], Expression],
) -> dict[int, Expression]:
    """Computes category probabilities for the ordered model, given thresholds.

    :param continuous_value: continuous value to map
    :param scale_parameter: scale parameter of the continuous value
    :param list_of_discrete_values: list of discrete values (length J)
    :param threshold_parameters: list of threshold Expressions (length J-1)
    :param cdf: CDF function
    :return: dict mapping discrete values to probabilities
    """
    if len(list_of_discrete_values) < 2:
        raise BiogemeError('Need at least two discrete values for ordered model.')
    if len(threshold_parameters) != len(list_of_discrete_values) - 1:
        raise BiogemeError(
            f'tau_parameters must have length len(list_of_discrete_values)-1, '
            f'got {len(threshold_parameters)} and {len(list_of_discrete_values)}.'
        )
    J = len(list_of_discrete_values)
    the_proba = {
        list_of_discrete_values[0]: 1
        - cdf((continuous_value - threshold_parameters[0]) / scale_parameter)
    }
    # First category
    # Middle categories
    for j in range(1, J - 1):
        the_proba[list_of_discrete_values[j]] = cdf(
            (continuous_value - threshold_parameters[j - 1]) / scale_parameter
        ) - cdf((continuous_value - threshold_parameters[j]) / scale_parameter)
    # Last category
    the_proba[list_of_discrete_values[-1]] = cdf(
        (continuous_value - threshold_parameters[-1]) / scale_parameter
    )
    return the_proba


def ordered_likelihood(
    continuous_value: Expression,
    scale_parameter: Expression,
    list_of_discrete_values: list[int],
    reference_threshold_parameter: Beta,
    cdf: Callable[[Expression], Expression],
) -> dict[int, Expression]:
    """Ordered model that maps a continuous quantity with a list of
    discrete intervals (often logit or probit).

    This function builds the ordered thresholds and computes the category probabilities.

    :param continuous_value: continuous value to mapping
    :param scale_parameter: scale parameter of the continuous value
    :param list_of_discrete_values: list of discrete values
    :param reference_threshold_parameter: parameter for the first threshold (Beta)
    :param cdf: function calculating the CDF of the random variable
    :return: dict mapping discrete values to probabilities
    """
    tau_parameters = build_ordered_thresholds(
        list_of_discrete_values, reference_threshold_parameter
    )
    return ordered_likelihood_from_thresholds(
        continuous_value, scale_parameter, list_of_discrete_values, tau_parameters, cdf
    )


def ordered_logit(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    reference_threshold_parameter: Beta,
) -> dict[int, Expression]:
    """Ordered logit model that maps a continuous quantity with a list of discrete intervals.

    This function builds the ordered thresholds and computes the category probabilities using the logistic CDF.
    """
    scale_parameter = validate_and_convert(scale_parameter)
    tau_parameters = build_ordered_thresholds(
        list_of_discrete_values, reference_threshold_parameter
    )
    return ordered_likelihood_from_thresholds(
        continuous_value,
        scale_parameter,
        list_of_discrete_values,
        tau_parameters,
        logisticcdf,
    )


def ordered_probit(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    reference_threshold_parameter: Beta,
) -> dict[int, Expression]:
    """Ordered probit model that maps a continuous quantity with a list of discrete intervals.

    This function builds the ordered thresholds and computes the category probabilities using the normal CDF.
    """
    scale_parameter = validate_and_convert(scale_parameter)
    tau_parameters = build_ordered_thresholds(
        list_of_discrete_values, reference_threshold_parameter
    )
    return ordered_likelihood_from_thresholds(
        continuous_value,
        scale_parameter,
        list_of_discrete_values,
        tau_parameters,
        NormalCdf,
    )


def ordered_logit_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
) -> dict[int, Expression]:
    """Ordered logit with explicit thresholds.

    Computes category probabilities given a list of thresholds. The number of
    thresholds must be one less than the number of categories.

    :param continuous_value: continuous value to map.
    :param scale_parameter: scale parameter of the continuous value
    :param list_of_discrete_values: ordered list of discrete categories (length ≥ 2).
    :param threshold_parameters: list of threshold expressions (length = len(list_of_discrete_values) - 1).
    :return: dict mapping each category to its probability.
    :raises BiogemeError: if lengths are incompatible.
    """
    return ordered_likelihood_from_thresholds(
        continuous_value,
        scale_parameter,
        list_of_discrete_values,
        threshold_parameters,
        logisticcdf,
    )


def ordered_probit_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
) -> dict[int, Expression]:
    """Ordered probit with explicit thresholds.

    Computes category probabilities given a list of thresholds. The number of
    thresholds must be one less than the number of categories.

    :param continuous_value: continuous value to map.
    :param scale_parameter: scale parameter of the continuous value

    :param list_of_discrete_values: ordered list of discrete categories (length ≥ 2).
    :param threshold_parameters: list of threshold expressions (length = len(list_of_discrete_values) - 1).
    :return: dict mapping each category to its probability.
    :raises BiogemeError: if lengths are incompatible.
    """
    return ordered_likelihood_from_thresholds(
        continuous_value,
        scale_parameter,
        list_of_discrete_values,
        threshold_parameters,
        NormalCdf,
    )
