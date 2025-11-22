"""Implements various models.

:author: Michel Bierlaire
:date: Fri Mar 29 17:13:14 2019
"""

import logging
from typing import Type

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression, validate_and_convert
from biogeme.expressions.ordered import (
    OrderedBase,
    OrderedLogLogit as OrderedLogLogitExpr,
    OrderedLogProbit as OrderedLogProbitExpr,
    OrderedLogit as OrderedLogitExpr,
    OrderedProbit as OrderedProbitExpr,
)

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


def _ordered_probs_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
    expr_cls: Type[OrderedBase],
) -> dict[int, Expression]:
    """Internal helper to build ordered logit/probit probabilities (or log-probabilities).

    The scale parameter is absorbed by dividing both the latent variable and
    the thresholds by the scale, so that the probability formula matches the
    standard ordered-response specification used in the original implementation.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param threshold_parameters: List of threshold expressions (length J-1).
    :param expr_cls: Ordered expression class to use
        (e.g. :class:`OrderedLogitExpr`, :class:`OrderedProbitExpr`,
        :class:`OrderedLogLogitExpr`, or :class:`OrderedLogProbitExpr`).
    :return: Dict mapping each category to its (log-)probability expression.
    :raises BiogemeError: If input lengths are inconsistent.
    """
    if len(list_of_discrete_values) < 2:
        raise BiogemeError("Need at least two discrete values for ordered model.")
    if len(threshold_parameters) != len(list_of_discrete_values) - 1:
        raise BiogemeError(
            "threshold_parameters must have length len(list_of_discrete_values)-1, "
            f"got {len(threshold_parameters)} and {len(list_of_discrete_values)}."
        )

    scale_expr = validate_and_convert(scale_parameter)

    # Rescaling: CDF is applied to (tau / sigma - eta / sigma),
    # matching the original formulas based on (x - tau) / sigma.
    eta_scaled: Expression = continuous_value / scale_expr
    cutpoints_scaled: list[Expression] = [
        tau / scale_expr for tau in threshold_parameters
    ]

    categories = list(list_of_discrete_values)
    probabilities: dict[int, Expression] = {}
    for cat in categories:
        # Constant response equal to this category; the Ordered* expression
        # then returns P(Y = cat) or log P(Y = cat) for each observation.
        y_expr = validate_and_convert(cat)
        model_expr = expr_cls(
            eta=eta_scaled,
            cutpoints=cutpoints_scaled,
            y=y_expr,
            categories=categories,
            neutral_labels=None,
            enforce_order=False,
        )
        probabilities[cat] = model_expr

    return probabilities


def ordered_logit(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    reference_threshold_parameter: Beta,
) -> dict[int, Expression]:
    """Ordered logit model that maps a continuous quantity with discrete intervals.

    This builds the ordered thresholds and returns per-category probabilities
    using the :class:`OrderedLogitExpr` expression.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param reference_threshold_parameter: Parameter for the first threshold (Beta).
    :return: Dict mapping each category to its probability expression.
    """
    tau_parameters = build_ordered_thresholds(
        list_of_discrete_values, reference_threshold_parameter
    )
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=tau_parameters,
        expr_cls=OrderedLogitExpr,
    )


def ordered_probit(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    reference_threshold_parameter: Beta,
) -> dict[int, Expression]:
    """Ordered probit model that maps a continuous quantity with discrete intervals.

    This builds the ordered thresholds and returns per-category probabilities
    using the :class:`OrderedProbitExpr` expression.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param reference_threshold_parameter: Parameter for the first threshold (Beta).
    :return: Dict mapping each category to its probability expression.
    """
    tau_parameters = build_ordered_thresholds(
        list_of_discrete_values, reference_threshold_parameter
    )
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=tau_parameters,
        expr_cls=OrderedProbitExpr,
    )


def ordered_logit_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
) -> dict[int, Expression]:
    """Ordered logit with explicit thresholds using :class:`OrderedLogitExpr`.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param threshold_parameters: List of threshold expressions (length J-1).
    :return: Dict mapping each category to its probability expression.
    """
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=threshold_parameters,
        expr_cls=OrderedLogitExpr,
    )


def ordered_probit_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
) -> dict[int, Expression]:
    """Ordered probit with explicit thresholds using :class:`OrderedProbitExpr`.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param threshold_parameters: List of threshold expressions (length J-1).
    :return: Dict mapping each category to its probability expression.
    """
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=threshold_parameters,
        expr_cls=OrderedProbitExpr,
    )


def log_ordered_logit(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    reference_threshold_parameter: Beta,
) -> dict[int, Expression]:
    """Log-ordered logit model that maps a continuous quantity with discrete intervals.

    This builds the ordered thresholds and returns per-category log-likelihood
    contributions (log-probabilities) using :class:`OrderedLogLogitExpr`.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param reference_threshold_parameter: Parameter for the first threshold (Beta).
    :return: Dict mapping each category to its log-probability expression.
    """
    tau_parameters = build_ordered_thresholds(
        list_of_discrete_values, reference_threshold_parameter
    )
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=tau_parameters,
        expr_cls=OrderedLogLogitExpr,
    )


def log_ordered_probit(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    reference_threshold_parameter: Beta,
) -> dict[int, Expression]:
    """Log-ordered probit model that maps a continuous quantity with discrete intervals.

    This builds the ordered thresholds and returns per-category log-likelihood
    contributions (log-probabilities) using :class:`OrderedLogProbitExpr`.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param reference_threshold_parameter: Parameter for the first threshold (Beta).
    :return: Dict mapping each category to its log-probability expression.
    """
    tau_parameters = build_ordered_thresholds(
        list_of_discrete_values, reference_threshold_parameter
    )
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=tau_parameters,
        expr_cls=OrderedLogProbitExpr,
    )


def log_ordered_logit_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
) -> dict[int, Expression]:
    """Log-ordered logit with explicit thresholds using :class:`OrderedLogLogitExpr`.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param threshold_parameters: List of threshold expressions (length J-1).
    :return: Dict mapping each category to its log-probability expression.
    """
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=threshold_parameters,
        expr_cls=OrderedLogLogitExpr,
    )


def log_ordered_probit_from_thresholds(
    continuous_value: Expression,
    scale_parameter: Expression | float,
    list_of_discrete_values: list[int],
    threshold_parameters: list[Expression],
) -> dict[int, Expression]:
    """Log-ordered probit with explicit thresholds using :class:`OrderedLogProbitExpr`.

    :param continuous_value: Continuous quantity to be mapped.
    :param scale_parameter: Scale parameter of the continuous value (sigma).
    :param list_of_discrete_values: Ordered list of discrete categories.
    :param threshold_parameters: List of threshold expressions (length J-1).
    :return: Dict mapping each category to its log-probability expression.
    """
    return _ordered_probs_from_thresholds(
        continuous_value=continuous_value,
        scale_parameter=scale_parameter,
        list_of_discrete_values=list_of_discrete_values,
        threshold_parameters=threshold_parameters,
        expr_cls=OrderedLogProbitExpr,
    )
