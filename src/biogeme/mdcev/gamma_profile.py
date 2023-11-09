"""Implementation of the "gamma profile" MDCEV model. See section 3.1 in
the technical report.

:author: Michel Bierlaire
:date: Sun Nov  5 15:56:36 2023

"""
from typing import Optional
from biogeme.expressions import Expression, log
from .mdcev import mdcev, info_gamma_parameters, SpecificModel


def gamma_profile(
    baseline_utilities: dict[int, Expression],
    consumed_quantities: dict[int, Expression],
    gamma_parameters: dict[int, Optional[Expression]],
    prices: Optional[dict[int, Expression]] = None,
):
    """Calculates the determinant entries for the linear expenditure system

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`
    :param prices: see the module documentation
        :mod:`biogeme.mdcev`. If None, assumed to be 1.

    """
    info_gamma_parameters(gamma_parameters)

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
            return baseline_utilities[the_id] - log(consumption)
        if price is None:
            return baseline_utilities[the_id] + log(gamma) - log(consumption + gamma)
        return (
            baseline_utilities[the_id] + log(gamma) - log(consumption + price * gamma)
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
            return -log(consumption)
        if price is None:
            return -log(consumption + gamma)
        return -log(consumption + price * gamma)

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
            return consumption
        if price is None:
            return consumption + gamma
        return consumption + price * gamma

    if prices:
        utilities = {
            the_id: (calculate_utility(the_id, consumption, prices[the_id]))
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
        the_id: calculate_inverse_determinant(the_id, consumption, None)
        for the_id, consumption in consumed_quantities.items()
    }
    return SpecificModel(
        utilities=utilities,
        log_determinant_entries=log_determinant_entries,
        inverse_of_determinant_entries=inverse_of_determinant_entries,
    )


def mdcev_gamma(
    number_of_chosen_alternatives: Expression,
    consumed_quantities: dict[int, Expression],
    baseline_utilities: dict[int, Expression],
    gamma_parameters: dict[int, Optional[Expression]],
    prices: Optional[dict[int, Expression]] = None,
    scale_parameter: Optional[Expression] = None,
):
    """Generate the Biogeme formula for the log probability of the
    MDCEV model using the linear expenditure system.

    :param number_of_chosen_alternatives: see the module documentation
        :mod:`biogeme.mdcev`
    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`
    :param gamma_parameters: see the module documentation
        :mod:`biogeme.mdcev`
    :param prices: see the module documentation :mod:`biogeme.mdcev`
    :param scale_parameter: see the module documentation :mod:`biogeme.mdcev`

    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    specific_model = gamma_profile(
        baseline_utilities,
        consumed_quantities,
        gamma_parameters,
        prices,
    )

    return mdcev(
        number_of_chosen_alternatives=number_of_chosen_alternatives,
        consumed_quantities=consumed_quantities,
        specific_model=specific_model,
        scale_parameter=scale_parameter,
    )
