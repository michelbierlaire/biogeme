"""Implements the probability generating function and the CDF of a
cross-nested logit model. This module is not used by Biogeme
itself. It is essentially used for external use. It has been mainly
implemented to verify the analytical derivatives of these functions.

:author: Michel Bierlaire
:date: Fri Apr 22 09:39:49 2022

"""
import numpy as np


def cnl_G(alternatives, nests):
    """Probability generating function and its derivatives

    :param alternatives: a list of alternatives in a given order. In
        principle, the alternative ids should be integers (to be
        consistent with Biogeme), but it may be actually be any object
        for this specific function.
    :type alternatives: list(int)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionary, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :return: function that calculates the G function, and its first
        and second derivatives.
    :rtype: f, g, H = fct(np.array(float))

    """
    order = {alt: index for index, alt in enumerate(alternatives)}
    J = len(alternatives)

    def G_and_deriv(y):
        """Probability generating function

        :param y: vector of positive values
        :type y: np.array

        :return: value of the CDF and its derivatives
        :rtype: floap, np.array(float), np.array(np.array(float))
        """

        G = 0.0
        Gi = np.zeros(J)
        Gij = np.zeros((J, J))
        for m in nests:
            mu_m = m[0]
            alphas = m[1]
            nest_specific_sum = 0.0
            for alpha_alt, alpha_value in alphas.items():
                if alpha_value != 0 and y[order[alpha_alt]] != 0:
                    nest_specific_sum += (
                        alpha_value * y[order[alpha_alt]]
                    ) ** mu_m
            p1 = (1.0 / mu_m) - 1.0
            p2 = (1.0 / mu_m) - 2.0
            G += nest_specific_sum ** (1.0 / mu_m)
            for i in range(J):
                alpha_i = alphas.get(alternatives[i], 0)
                if alpha_i != 0 and y[i] != 0:
                    Gi[i] += (
                        alpha_i**mu_m
                        * y[i] ** (mu_m - 1)
                        * nest_specific_sum**p1
                    )
                    Gij[i][i] += (
                        1 - mu_m
                    ) * nest_specific_sum**p2 * alpha_i ** (2 * mu_m) * y[
                        i
                    ] ** (
                        2 * mu_m - 2.0
                    ) + (
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


def cnl_CDF(alternatives, nests):
    """Cumulative distribution function and its derivatives

    :param alternatives: a list of alternatives in a given order. In
        principle, the alternative ids should be integers (to be
        consistent with Biogeme), but it may be actually be any object
        for this specific function.
    :type alternatives: list(int)

    :param nests: a tuple containing as many items as nests.
        Each item is also a tuple containing two items:

        - an object of type biogeme.expressions. expr.Expression
          representing the nest parameter,
        - a dictionary mapping the alternative ids with the cross-nested
          parameters for the corresponding nest. If an alternative is
          missing in the dictionaray, the corresponding alpha is set to zero.

        Example::

            alphaA = {1: alpha1a,
                      2: alpha2a,
                      3: alpha3a,
                      4: alpha4a,
                      5: alpha5a,
                      6: alpha6a}
            alphaB = {1: alpha1b,
                      2: alpha2b,
                      3: alpha3b,
                      4: alpha4b,
                      5: alpha5b,
                      6: alpha6b}
            nesta = MUA, alphaA
            nestb = MUB, alphaB
            nests = nesta, nestb

    :type nests: tuple

    :return: function that calculates the CDF, and its first
        and second derivatives.
    :rtype: f, g, H = fct(np.array(float))

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
            Fij[i][i] = (
                F * y[i] * y[i] * (Gi[i] * Gi[i] - Gii[i][i])
                - F * Gi[i] * y[i]
            )
            for j in range(i + 1, J):
                Fij[i][j] = F * y[i] * y[j] * (Gi[i] * Gi[j] - Gii[i][j])
        for i in range(J):
            for j in range(i + 1, J):
                Fij[j][i] = Fij[i][j]

        return F, Fi, Fij

    return F_and_deriv
