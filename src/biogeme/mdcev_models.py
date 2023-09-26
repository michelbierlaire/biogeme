"""Implementation of the various versions of the MDCEV model

Refer to the techical report for the mathematical expressions implemented in this file.
The report mentions three models:

 - translated utility function: labeled "translated" in this
   implementation.
 - generalized translated utility function: labeled "geneneralized" in
   this implementation
 - linear expenditure system: labeled "gamma_profile" in this implementation

:author: Michel Bierlaire
:date: Thu Aug 24 09:14:36 2023

"""
from typing import NamedTuple
from biogeme.expressions import Expression, log, exp


class SpecificModel(NamedTuple):
    utilities: dict[int, Expression]
    log_determinant_entries: dict[int, Expression]
    inverse_of_determinant_entries: dict[int, Expression]


def translated(
    baseline_utilities,
    consumed_quantities,
    alpha_parameters,
    gamma_parameters,
):
    """Calculates

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type alpha_parameters: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    """
    utilities = {
        id: (
            baseline_utilities[id]
            + log(alpha_parameters[id])
            + (alpha_parameters[id] - 1) * log(consumption + gamma_parameters[id])
        )
        for id, consumption in consumed_quantities.items()
    }

    log_determinant_entries = {
        id: log(1 - alpha_parameters[id]) - log(consumption + gamma_parameters[id])
        for id, consumption in consumed_quantities.items()
    }

    inverse_of_determinant_entries = {
        id: (consumption + gamma_parameters[id]) / (1 - alpha_parameters[id])
        for id, consumption in consumed_quantities.items()
    }

    return SpecificModel(
        utilities=utilities,
        log_determinant_entries=log_determinant_entries,
        inverse_of_determinant_entries=inverse_of_determinant_entries,
    )


def generalized(
    baseline_utilities,
    consumed_quantities,
    alpha_parameters,
    gamma_parameters,
    prices=None,
):
    """Calculates the determinant entries for the Bhat 2008 specification Eq (18)

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type alpha_parameters: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    :param prices: see the module documentation :mod:`biogeme.mdcev`.
        If None, assumed to be 1.
    :type prices: biogeme.expressions.Expression

    """
    if prices:
        utilities = {
            id: (
                baseline_utilities[id]
                - log(prices[id])
                + +(alpha_parameters[id] - 1)
                * log(1.0 + consumption / (prices[id] * gamma_parameters[id]))
            )
            for id, consumption in consumed_quantities.items()
        }
        log_determinant_entries = {
            id: log(1 - alpha_parameters[id])
            - log(consumption + prices[id] * gamma_parameters[id])
            for id, consumption in consumed_quantities.items()
        }
        inverse_of_determinant_entries = {
            id: (consumption + prices[id] * gamma_parameters[id])
            / (1 - alpha_parameters[id])
            for id, consumption in consumed_quantities.items()
        }
        return SpecificModel(
            utilities=utilities,
            log_determinant_entries=log_determinant_entries,
            inverse_of_determinant_entries=inverse_of_determinant_entries,
        )
    # Use unit prices if prices is set to None
    utilities = {
        id: (
            baseline_utilities[id]
            + (alpha_parameters[id] - 1) * log(1.0 + consumption / gamma_parameters[id])
        )
        for id, consumption in consumed_quantities.items()
    }
    log_determinant_entries = {
        id: log(1 - alpha_parameters[id]) - log(consumption + gamma_parameters[id])
        for id, consumption in consumed_quantities.items()
    }
    inverse_of_determinant_entries = {
        id: (consumption + gamma_parameters[id]) / (1 - alpha_parameters[id])
        for id, consumption in consumed_quantities.items()
    }
    return SpecificModel(
        utilities=utilities,
        log_determinant_entries=log_determinant_entries,
        inverse_of_determinant_entries=inverse_of_determinant_entries,
    )


def gamma_profile(
    baseline_utilities,
    consumed_quantities,
    gamma_parameters,
    prices=None,
):
    """Calculates the determinant entries for the linear expenditure system

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    :param prices: see the module documentation
        :mod:`biogeme.mdcev`. If None, assumed to be 1.
    :type prices: biogeme.expressions.Expression


    """
    if prices:
        utilities = {
            id: (
                baseline_utilities[id]
                + log(gamma_parameters[id])
                - log(consumption + prices[id] * gamma_parameters[id])
            )
            for id, consumption in consumed_quantities.items()
        }
        log_determinant_entries = {
            id: -log(consumption + prices[id] * gamma_parameters[id])
            for id, consumption in consumed_quantities.items()
        }
        inverse_of_determinant_entries = {
            id: consumption + prices[id] * gamma_parameters[id]
            for id, consumption in consumed_quantities.items()
        }
        return SpecificModel(
            utilities=utilities,
            log_determinant_entries=log_determinant_entries,
            inverse_of_determinant_entries=inverse_of_determinant_entries,
        )

    utilities = {
        id: (
            baseline_utilities[id]
            + log(gamma_parameters[id])
            - log(consumption + gamma_parameters[id])
        )
        for id, consumption in consumed_quantities.items()
    }
    log_determinant_entries = {
        id: -log(consumption + gamma_parameters[id])
        for id, consumption in consumed_quantities.items()
    }
    inverse_of_determinant_entries = {
        id: consumption + gamma_parameters[id]
        for id, consumption in consumed_quantities.items()
    }
    return SpecificModel(
        utilities=utilities,
        log_determinant_entries=log_determinant_entries,
        inverse_of_determinant_entries=inverse_of_determinant_entries,
    )


def non_monotonic(
    baseline_utilities,
    mu_utilities,
    consumed_quantities,
    alpha_parameters,
    gamma_parameters,
    prices=None,
):
    """Calculates the determinant entries for the linear expenditure system

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :type baseline_utilities: dict[int: biogeme.expression.Expression]

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :type consumed_quantities: dict[int: biogeme.expression.Expression]

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type alpha_parameters: dict[int: biogeme.expression.Expression]

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :type gamma_parameters: dict[int: biogeme.expression.Expression]

    :param prices: see the module documentation
        :mod:`biogeme.mdcev`. If None, assumed to be 1.
    :type prices: biogeme.expressions.Expression


    """
    if prices:
        utilities = {
            id: (
                mu_utilities[id]
                + exp(baseline_utilities[id])
                * (1 + consumption / (prices[id] * gamma_parameters[id]))
                ** (alpha_parameters[id] - 1)
                / prices[id]
            )
            for id, consumption in consumed_quantities.items()
        }

        log_determinant_entries = {
            id: -2 * log(prices[id])
            + baseline_utilities[id]
            + log(1 - alpha_parameters[id])
            - log(gamma_parameters[id])
            + (alpha_parameters[id] - 2)
            * log(1 + consumption / (prices[id] * gamma_parameters[id]))
            for id, consumption in consumed_quantities.items()
        }

        inverse_of_determinant_entries = {
            id: prices[id]
            * prices[id]
            * exp(-baseline_utilities[id])
            * gamma_parameters[id]
            * (1 + consumption / (prices[id] * gamma_parameters[id]))
            ** (2 - alpha_parameters[id])
            / (1 - alpha_parameters[id])
            for id, consumption in consumed_quantities.items()
        }
        return SpecificModel(
            utilities=utilities,
            log_determinant_entries=log_determinant_entries,
            inverse_of_determinant_entries=inverse_of_determinant_entries,
        )

    utilities = {
        id: (
            mu_utilities[id]
            + exp(baseline_utilities[id])
            * (1 + consumption / gamma_parameters[id]) ** (alpha_parameters[id] - 1)
        )
        for id, consumption in consumed_quantities.items()
    }

    log_determinant_entries = {
        id: baseline_utilities[id]
        + log(1 - alpha_parameters[id])
        - log(gamma_parameters[id])
        + (alpha_parameters[id] - 2) * log(1 + consumption / gamma_parameters[id])
        for id, consumption in consumed_quantities.items()
    }

    inverse_of_determinant_entries = {
        id: exp(-baseline_utilities[id])
        * gamma_parameters[id]
        * (1 + consumption / gamma_parameters[id]) ** (2 - alpha_parameters[id])
        / (1 - alpha_parameters[id])
        for id, consumption in consumed_quantities.items()
    }
    return SpecificModel(
        utilities=utilities,
        log_determinant_entries=log_determinant_entries,
        inverse_of_determinant_entries=inverse_of_determinant_entries,
    )
