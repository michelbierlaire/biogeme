"""Implements the probability generating function and the CDF of a
cross-nested logit model. This module is not used by Biogeme
itself. It is essentially used for external use. It has been mainly
implemented to verify the analytical derivatives of these functions.

:author: Michel Bierlaire
:date: Fri Apr 22 09:39:49 2022

"""

from typing import Callable
import numpy as np

from biogeme.expressions import get_dict_values
from biogeme.function_output import FunctionOutput
from biogeme.nests import NestsForCrossNestedLogit
from biogeme.deprecated import deprecated


def cnl_g(
    alternatives: list[int], nests: NestsForCrossNestedLogit
) -> Callable[[np.ndarray], FunctionOutput]:
    """Probability generating function and its derivatives

    :param alternatives: a list of alternatives in a given order. In
        principle, the alternative ids should be integers (to be
        consistent with Biogeme), but it may be actually be any object
        for this specific function.

    :param nests: the object describing the nests.

    :return: function that calculates the G function, and its first
        and second derivatives.
    :rtype: f, g, H = fct(np.array(float))

    """
    order = {alt: index for index, alt in enumerate(alternatives)}
    nbr_of_alternatives = len(alternatives)

    def g_and_deriv(y: np.ndarray) -> FunctionOutput:
        """Probability generating function

        :param y: vector of positive values

        :return: value of the CDF and its derivatives
        :rtype: floap, np.array(float), np.array(np.array(float))
        """

        g = 0.0
        g_i = np.zeros(nbr_of_alternatives)
        g_ij = np.zeros((nbr_of_alternatives, nbr_of_alternatives))
        for m in nests:
            mu_m = m.nest_param
            alphas = get_dict_values(m.dict_of_alpha)
            nest_specific_sum = 0.0
            for alpha_alt, alpha_value in alphas.items():
                if alpha_value != 0 and y[order[alpha_alt]] != 0:
                    nest_specific_sum += (alpha_value * y[order[alpha_alt]]) ** mu_m
            p1 = (1.0 / mu_m) - 1.0
            p2 = (1.0 / mu_m) - 2.0
            g += nest_specific_sum ** (1.0 / mu_m)
            for i in range(nbr_of_alternatives):
                alpha_i = alphas.get(alternatives[i], 0)
                if alpha_i != 0 and y[i] != 0:
                    g_i[i] += alpha_i**mu_m * y[i] ** (mu_m - 1) * nest_specific_sum**p1
                    g_ij[i][i] += (1 - mu_m) * nest_specific_sum**p2 * alpha_i ** (
                        2 * mu_m
                    ) * y[i] ** (2 * mu_m - 2.0) + (
                        mu_m - 1
                    ) * nest_specific_sum**p1 * alpha_i**mu_m * y[
                        i
                    ] ** (
                        mu_m - 2
                    )
                    for j in range(i + 1, nbr_of_alternatives):
                        alpha_j = alphas.get(alternatives[j], 0)
                        if alpha_j != 0 and y[j] != 0:
                            g_ij[i][j] += (
                                (1 - mu_m)
                                * nest_specific_sum**p2
                                * (alpha_i * alpha_j) ** mu_m
                                * (y[i] * y[j]) ** (mu_m - 1.0)
                            )

        for i in range(nbr_of_alternatives):
            for j in range(i + 1, nbr_of_alternatives):
                g_ij[j][i] = g_ij[i][j]

        return FunctionOutput(function=g, gradient=g_i, hessian=g_ij)

    return g_and_deriv


@deprecated(new_func=cnl_g)
def cnl_G(
    alternatives: list[int], nests: NestsForCrossNestedLogit
) -> Callable[[np.ndarray], FunctionOutput]:
    pass


def cnl_cdf(
    alternatives: list[int], nests: NestsForCrossNestedLogit
) -> Callable[[np.ndarray], FunctionOutput]:
    """Cumulative distribution function and its derivatives

    :param alternatives: a list of alternatives in a given order. In
        principle, the alternative ids should be integers (to be
        consistent with Biogeme), but it may be actually be any object
        for this specific function.

    :param nests: a tuple containing as many items as nests.

    :return: function that calculates the CDF, and its first
        and second derivatives.

    """
    nbr_of_alternatives = len(alternatives)

    g_fct = cnl_g(alternatives, nests)

    def f_and_deriv(xi: np.ndarray) -> FunctionOutput:
        """Cumulative distribution function

        :param xi: vector of arguments
        :type xi: np.array(float)

        :return: value of the CDF and its derivatives
        :rtype: float, np.array(float), np.array(np.array(float))
        """
        y = np.where(xi == np.inf, 0, np.exp(-xi))
        g_output: FunctionOutput = g_fct(y)
        g = g_output.function
        g_i = g_output.gradient
        g_ii = g_output.hessian

        f = np.exp(-g)
        f_i = g_i * y * f
        f_ij = np.zeros((nbr_of_alternatives, nbr_of_alternatives))

        for i in range(nbr_of_alternatives):
            f_ij[i][i] = (
                f * y[i] * y[i] * (g_i[i] * g_i[i] - g_ii[i][i]) - f * g_i[i] * y[i]
            )
            for j in range(i + 1, nbr_of_alternatives):
                f_ij[i][j] = f * y[i] * y[j] * (g_i[i] * g_i[j] - g_ii[i][j])
        for i in range(nbr_of_alternatives):
            for j in range(i + 1, nbr_of_alternatives):
                f_ij[j][i] = f_ij[i][j]

        return FunctionOutput(function=f, gradient=f_i, hessian=f_ij)

    return f_and_deriv


@deprecated(new_func=cnl_cdf)
def cnl_CDF(
    alternatives: list[int], nests: NestsForCrossNestedLogit
) -> Callable[[np.ndarray], FunctionOutput]:
    pass
