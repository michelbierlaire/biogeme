"""Implementation of the "non monotonic" MDCEV model. See section 3.2 in
the technical report.

:author: Michel Bierlaire
:date: Sun Nov  5 15:58:46 2023

"""
from typing import Optional
from biogeme.expressions import Expression, log, exp, Numeric
from .mdcev import mdcev, info_gamma_parameters, SpecificModel


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
    info_gamma_parameters()

    def calculate_utility(
        the_id: int, consumption: Expression, price: Optional[Expression]
    ) -> Expression:
        """Calculate the utility. The formula is different is it is an
            outside good, characterized by the absence of a gamma
            parameter.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        :param price: expression for the price, or None if prices are not considered.
        """

        gamma = gamma_parameters[the_id]
        if gamma is None:
            if price is None:
                # gamma is None. price is None.
                return mu_utilities[the_id] + exp(
                    baseline_utilities[the_id]
                ) * consumption ** (alpha_parameters[the_id] - 1)
            # gamma is None. price is not None.
            return (
                mu_utilities[the_id]
                + exp(baseline_utilities[the_id])
                * (consumption / price) ** (alpha_parameters[the_id] - 1)
                / price
            )
        if price is None:
            # gamma is not None. price is None.
            return mu_utilities[the_id] + exp(baseline_utilities[the_id]) * (
                1 + consumption / gamma
            ) ** (alpha_parameters[the_id] - 1)
        # gamma is not None. price is not None.
        return (
            mu_utilities[the_id]
            + exp(baseline_utilities[the_id])
            * (1 + consumption / (price * gamma)) ** (alpha_parameters[the_id] - 1)
            / price
        )

    def calculate_log_determinant(
        the_id: int, consumption: Expression, price: Optional[Expression]
    ) -> Expression:
        """Calculate the log of the entries for the determinant. For
            the outside good, gamma is equal to 0.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        :param price: expression for the price, or None if prices are not considered.
        """
        gamma = gamma_parameters[the_id]
        if gamma is None:
            if price is None:
                # gamma is None. price is None.
                return (
                    baseline_utilities[the_id]
                    + log(1 - alpha_parameters[the_id])
                    + (alpha_parameters[the_id] - 2) * log(consumption)
                )
                # gamma is None. price is not None.
                return (
                    baseline_utilities[the_id]
                    + log(1 - alpha_parameters[the_id])
                    + (alpha_parameters[the_id] - 2) * log(consumption)
                    - alpha_parameters[the_id] * log(price)
                )
        if price is None:
            # gamma is not None. price is None
            return (
                baseline_utilities[the_id]
                + log(1 - alpha_parameters[the_id])
                - log(gamma)
                + (alpha_parameters[the_id] - 2) * log(1 + consumption / gamma)
            )
        # gamma is not None. price is not None
        return (
            Numeric(-2) * log(price)
            + baseline_utilities[the_id]
            + log(Numeric(1) - alpha_parameters[the_id])
            - log(gamma)
            + (alpha_parameters[the_id] - Numeric(2))
            * log(Numeric(1) + consumption / (price * gamma))
        )

    def calculate_inverse_determinant(
        the_id: int, consumption: Expression, price: Optional[Expression]
    ) -> Expression:
        """Calculate the inverse of the entries for the determinant. For
            the outside good, gamma is equal to 0.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        :param price: expression for the price, or None if prices are not considered.
        """
        gamma = gamma_parameters[the_id]
        if gamma is None:
            if price is None:
                # gamma is None. price is None.
                return (
                    exp(-baseline_utilities[the_id])
                    * consumption ** (2 - alpha_parameters[the_id])
                    / (1 - alpha_parameters[the_id])
                )
            # gamma is None. price is not None.
            return (
                price
                * price
                * exp(-baseline_utilities[the_id])
                * (consumption / price) ** (2 - alpha_parameters[the_id])
                / (1 - alpha_parameters[the_id])
            )
        if price is None:
            # gamma is not None. price is None.
            return (
                exp(-baseline_utilities[the_id])
                * gamma
                * (1 + consumption / gamma) ** (2 - alpha_parameters[the_id])
                / (1 - alpha_parameters[the_id])
            )
        # gamma is not None. price is not None.
        return (
            price
            * price
            * exp(-baseline_utilities[the_id])
            * gamma
            * (1 + consumption / (price * gamma)) ** (2 - alpha_parameters[the_id])
            / (1 - alpha_parameters[the_id])
        )

    if prices:
        utilities = {
            the_id: calculate_utility(the_id, consumption, prices[the_id])
            for the_id, consumption in consumed_quantities.items()
        }

        log_determinant_entries = {
            the_id: calculate_log_determinant(the_id, consumption, prices[the_id])
            for the_id, consumption in consumed_quantities.items()
        }

        inverse_of_determinant_entries = {
            the_id: calculate_inverse_determinant(the_id, consumption, prices[the_id])
            for the_id, consumption in consumed_quantities.items()
        }
        return SpecificModel(
            utilities=utilities,
            log_determinant_entries=log_determinant_entries,
            inverse_of_determinant_entries=inverse_of_determinant_entries,
        )

    utilities = {
        the_id: calculate_utility(the_id, consumption, None)
        for the_id, consumption in consumed_quantities.items()
    }

    log_determinant_entries = {
        the_id: calculate_log_determinant(the_id, consumption, None)
        for the_id, consumption in consumed_quantities.items()
    }

    inverse_of_determinant_entries = {
        id: calculate_inverse_determinant(the_id, consumption, None)
        for the_id, consumption in consumed_quantities.items()
    }
    return SpecificModel(
        utilities=utilities,
        log_determinant_entries=log_determinant_entries,
        inverse_of_determinant_entries=inverse_of_determinant_entries,
    )


def mdcev_non_monotonic(
    number_of_chosen_alternatives: Expression,
    consumed_quantities: dict[int, Expression],
    baseline_utilities: dict[int, Expression],
    mu_utilities: dict[int, Expression],
    alpha_parameters: dict[int, Expression],
    gamma_parameters: dict[int, Optional[Expression]],
    prices: Optional[dict[int, Expression]] = None,
):
    """Generate the Biogeme formula for the log probability of the
    MDCEV model using the linear expenditure system.

    :param number_of_chosen_alternatives: see the module documentation
        :mod:`biogeme.mdcev`

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`

    :param mu_utilities: additional utility for the non-monotonic
        part. see the module documentation :mod:`biogeme.mdcev`

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`

    :param prices: see the module documentation :mod:`biogeme.mdcev`

    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    specific_model = non_monotonic(
        baseline_utilities,
        mu_utilities,
        consumed_quantities,
        alpha_parameters,
        gamma_parameters,
        prices,
    )

    return mdcev(
        number_of_chosen_alternatives=number_of_chosen_alternatives,
        consumed_quantities=consumed_quantities,
        specific_model=specific_model,
        scale_parameter=None,
    )
