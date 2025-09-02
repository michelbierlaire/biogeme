"""Calculates the contribution to the likelihood function of an observation in a linear regression context.

Michel Bierlaire
Sun Aug 17 2025, 18:18:23
"""

from biogeme.distributions import lognormalpdf, normal_logpdf, normalpdf
from biogeme.expressions import (
    Beta,
    Expression,
    LinearTermTuple,
    LinearUtility,
    Variable,
)


def build_linear_terms(
    independent_variables: list[Variable], coefficients: list[Beta]
) -> LinearUtility:
    if len(independent_variables) != len(coefficients):
        raise ValueError(
            f'There are {len(independent_variables)} variables and {len(coefficients)} coefficients. Thi sis inconsistent.'
        )
    return LinearUtility(
        [
            LinearTermTuple(beta=beta, x=x)
            for beta, x in zip(coefficients, independent_variables)
        ]
    )


def build_normalized_formula(
    dependent_variable: Expression,
    linear_terms: Expression,
    scale_parameter: Expression,
) -> Expression:
    """
    Constructs the standardized residual expression used in the likelihood and loglikelihood.

    :param dependent_variable: The dependent variable expression.
    :param linear_terms: Expression for the linear terms.
    :param scale_parameter: The scale parameter expression (e.g., standard deviation).
    :return: An Expression representing the standardized residual (dependent_variable minus linear predictor, divided by scale).
    """
    return (dependent_variable - linear_terms) / scale_parameter


def regression_likelihood(
    dependent_variable: Expression,
    linear_terms: Expression,
    scale_parameter: Expression,
) -> Expression:
    """
    Calculates the contribution of one observation to the likelihood under a normal regression model.

    :param dependent_variable: The dependent variable expression.
    :param linear_terms: Expression for the linear terms.
    :param scale_parameter: The scale parameter expression (e.g., standard deviation).
    :return: An Expression representing the likelihood contribution of the observation.
    """
    argument = build_normalized_formula(
        dependent_variable=dependent_variable,
        linear_terms=linear_terms,
        scale_parameter=scale_parameter,
    )
    return normalpdf(argument)


def regression_loglikelihood(
    dependent_variable: Expression,
    linear_terms: Expression,
    scale_parameter: Expression,
) -> Expression:
    """
    Calculates the log-likelihood contribution of one observation under a normal regression model.

    :param dependent_variable: The dependent variable expression.
    :param linear_terms: Expression for the linear terms.
    :param scale_parameter: The scale parameter expression (e.g., standard deviation).
    :return: An Expression representing the log-likelihood contribution of the observation.
    """
    argument = build_normalized_formula(
        dependent_variable=dependent_variable,
        linear_terms=linear_terms,
        scale_parameter=scale_parameter,
    )
    return normal_logpdf(argument)
