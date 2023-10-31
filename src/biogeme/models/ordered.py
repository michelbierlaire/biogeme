""" Implements various models.

:author: Michel Bierlaire
:date: Fri Mar 29 17:13:14 2019
"""
import logging
from typing import Callable, Mapping
import biogeme.distributions as dist
import biogeme.exceptions as excep
from biogeme.expressions import Expression, Beta, bioNormalCdf

logger = logging.getLogger(__name__)


def ordered_likelihood(
    continuous_value: Expression,
    list_of_discrete_values: list[int],
    tau_parameter: Beta,
    cdf: Callable[[Expression], Expression],
) -> Mapping[int, Expression]:
    """Ordered model that maps a continuous quantity with a list of
        discrete intervals (often logit or probit)

    Example: discrete values = [1, 2, 3, 4]

    We define thresholds tau_1_2, tau_2_3 and tau_3_4.
    In order to impose that the threshold are sorted, we actually define
        tau_1_2 = tau_parameter
        tau_2_3 = tau_1_2 + diff2
        tau_3_4 = tau_2_3 + diff3

    The probability that the discrete value is 2, say, is the
    probability that the continuous value lies between tau_1_2 and
    tau_2_3, where the probability distribution is logistic.

    :param continuous_value: continuous quantity to mapping

    :param list_of_discrete_values: discrete values

    :param tau_parameter: parameter for the first threshold

    :param cdf: function calculating the CDF of the random variable

    :return: dict where the keys are the discrete values and the
        values are the corresponding probability.
    """
    if not isinstance(tau_parameter, Beta):
        error_msg = (
            f'tau_parameter must be a Beta expression, and not a {type(tau_parameter)}.'
        )
        raise excep.BiogemeError(error_msg)

    if len(list_of_discrete_values) == 2:
        the_proba = {
            list_of_discrete_values[0]: 1 - cdf(continuous_value - tau_parameter),
            list_of_discrete_values[1]: cdf(continuous_value - tau_parameter),
        }

        return the_proba

    diffs = {
        current_item: Beta(
            f'{tau_parameter.name}_diff_{current_item}',
            1,
            0,
            None,
            0,
        )
        for current_item in list_of_discrete_values[1:-1]
    }

    # First term
    the_proba = {list_of_discrete_values[0]: 1 - cdf(continuous_value - tau_parameter)}

    # Intermediate terms
    tau = tau_parameter
    for item in list_of_discrete_values[1:-1]:
        next_tau = tau + diffs[item]
        the_proba[item] = cdf(continuous_value - tau) - cdf(continuous_value - next_tau)
        tau = next_tau

    # Last term
    the_proba[list_of_discrete_values[-1]] = cdf(continuous_value - tau)

    return the_proba


def ordered_logit(
    continuous_value: Expression,
    list_of_discrete_values: list[int],
    tau_parameter: Beta,
) -> Mapping[int, Expression]:
    """Ordered logit model that maps a continuous quantity with a
        list of discrete intervals

    Example: discrete values = [1, 2, 3, 4]

    We define thresholds tau_1_2, tau_2_3 and tau_3_4.
    In order to impose that the threshold are sorted, we actually define
        tau_1_2 = tau_parameter
        tau_2_3 = tau_1_2 + diff2
        tau_3_4 = tau_2_3 + diff3

    The probability that the discrete value is 2, say, is the
    probability that the continuous value lies between tau_1_2 and
    tau_2_3, where the probability distribution is logistic.

    :param continuous_value: continuous quantity to mapping

    :param list_of_discrete_values: discrete values

    :param tau_parameter: parameter for the first threshold

    """
    return ordered_likelihood(
        continuous_value=continuous_value,
        list_of_discrete_values=list_of_discrete_values,
        tau_parameter=tau_parameter,
        cdf=dist.logisticcdf,
    )


def ordered_probit(
    continuous_value: Expression,
    list_of_discrete_values: list[int],
    tau_parameter: Beta,
) -> Mapping[int, Expression]:
    """Ordered probit model that maps a continuous quantity with a
        list of discrete intervals

    Example: discrete values = [1, 2, 3, 4]

    We define thresholds tau_1_2, tau_2_3 and tau_3_4.
    In order to impose that the threshold are sorted, we actually define
        tau_1_2 = tau_parameter
        tau_2_3 = tau_1_2 + diff2
        tau_3_4 = tau_2_3 + diff3

    The probability that the discrete value is 2, say, is the
    probability that the continuous value lies between tau_1_2 and
    tau_2_3, where the probability distribution is normal.

    :param continuous_value: continuous quantity to mapping

    :param list_of_discrete_values: discrete values

    :param tau_parameter: parameter for the first threshold

    """
    return ordered_likelihood(
        continuous_value=continuous_value,
        list_of_discrete_values=list_of_discrete_values,
        tau_parameter=tau_parameter,
        cdf=bioNormalCdf,
    )
