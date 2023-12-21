"""Implements the probability generating function and the CDF of a
cross-nested logit model. This module is not used by Biogeme
itself. It is essentially used for external use. It has been mainly
implemented to verify the analytical derivatives of these functions.

:author: Michel Bierlaire
:date: Fri Apr 22 09:39:49 2022

"""
from typing import Callable
import numpy as np
from biogeme.nests import NestsForCrossNestedLogit, get_alpha_values


def cnl_G(
    alternatives: list[int], nests: NestsForCrossNestedLogit
) -> Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]:
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
    J = len(alternatives)

    def G_and_deriv(y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Probability generating function

        :param y: vector of positive values

        :return: value of the CDF and its derivatives
        :rtype: floap, np.array(float), np.array(np.array(float))
        """

        G = 0.0
        Gi = np.zeros(J)
        Gij = np.zeros((J, J))
        for m in nests:
            mu_m = m.nest_param
            alphas = get_alpha_values(m.dict_of_alpha)
            nest_specific_sum = 0.0
            for alpha_alt, alpha_value in alphas.items():
                if alpha_value != 0 and y[order[alpha_alt]] != 0:
                    nest_specific_sum += (alpha_value * y[order[alpha_alt]]) ** mu_m
            p1 = (1.0 / mu_m) - 1.0
            p2 = (1.0 / mu_m) - 2.0
            G += nest_specific_sum ** (1.0 / mu_m)
            for i in range(J):
                alpha_i = alphas.get(alternatives[i], 0)
                if alpha_i != 0 and y[i] != 0:
                    Gi[i] += (
                        alpha_i**mu_m * y[i] ** (mu_m - 1) * nest_specific_sum**p1
                    )
                    Gij[i][i] += (1 - mu_m) * nest_specific_sum**p2 * alpha_i ** (
                        2 * mu_m
                    ) * y[i] ** (2 * mu_m - 2.0) + (
                        mu_m - 1
                    ) * nest_specific_sum**p1 * alpha_i**mu_m * y[
                        i
                    ] ** (
                        mu_m - 2
                    )
                    for j in range(i + 1, J):
                        alpha_j = alphas.get(alternatives[j], 0)
                        if alpha_j != 0 and y[j] != 0:
                            Gij[i][j] += (
                                (1 - mu_m)
                                * nest_specific_sum**p2
                                * (alpha_i * alpha_j) ** mu_m
                                * (y[i] * y[j]) ** (mu_m - 1.0)
                            )

        for i in range(J):
            for j in range(i + 1, J):
                Gij[j][i] = Gij[i][j]

        return G, Gi, Gij

    return G_and_deriv


def cnl_CDF(
    alternatives: list[int], nests: NestsForCrossNestedLogit
) -> Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]:
    """Cumulative distribution function and its derivatives

    :param alternatives: a list of alternatives in a given order. In
        principle, the alternative ids should be integers (to be
        consistent with Biogeme), but it may be actually be any object
        for this specific function.

    :param nests: a tuple containing as many items as nests.

    :return: function that calculates the CDF, and its first
        and second derivatives.

    """
    J = len(alternatives)

    G_fct = cnl_G(alternatives, nests)

    def F_and_deriv(xi):
        """Cumulative distribution function

        :param xi: vector of arguments
        :type xi: np.array(float)

        :return: value of the CDF and its derivatives
        :rtype: floap, np.array(float), np.array(np.array(float))
        """
        y = np.where(xi == np.inf, 0, np.exp(-xi))
        G, Gi, Gii = G_fct(y)

        F = np.exp(-G)
        Fi = Gi * y * F
        Fij = np.zeros((J, J))

        for i in range(J):
            Fij[i][i] = F * y[i] * y[i] * (Gi[i] * Gi[i] - Gii[i][i]) - F * Gi[i] * y[i]
            for j in range(i + 1, J):
                Fij[i][j] = F * y[i] * y[j] * (Gi[i] * Gi[j] - Gii[i][j])
        for i in range(J):
            for j in range(i + 1, J):
                Fij[j][i] = Fij[i][j]

        return F, Fi, Fij

    return F_and_deriv
