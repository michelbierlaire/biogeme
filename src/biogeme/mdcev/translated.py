"""Implementation of the "translated" MDCEV model. See section 2.1 in
the technical report.

:author: Michel Bierlaire
:date: Sun Nov  5 15:43:18 2023

"""
from typing import Optional
from biogeme.expressions import Expression, log, Numeric

from .mdcev import mdcev, info_gamma_parameters, SpecificModel


def translated(
    baseline_utilities: dict[int, Expression],
    consumed_quantities: dict[int, Expression],
    alpha_parameters: dict[int, Expression],
    gamma_parameters: dict[int, Optional[Expression]],
):
    """Calculates

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`

    :param consumed_quantities: see the module documentation
        :mod:`biogeme.mdcev`

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`

    """
    info_gamma_parameters(gamma_parameters)

    def calculate_utility(the_id: int, consumption: Expression) -> Expression:
        """Calculate the utility. The formula is different is it is an
            outside good, characterized by the absence of a gamma
            parameter.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        :param price: expression for the price, or None if prices are not considered.
        """
        gamma = gamma_parameters[the_id]
        if gamma is None:
            return (
                baseline_utilities[the_id]
                + log(alpha_parameters[the_id])
                + (alpha_parameters[the_id] - Numeric(1)) * log(consumption)
            )
        return (
            baseline_utilities[the_id]
            + log(alpha_parameters[the_id])
            + (alpha_parameters[the_id] - Numeric(1)) * log(consumption + gamma)
        )

    def calculate_log_determinant(the_id: int, consumption: Expression) -> Expression:
        """Calculate the log of the entries for the determinant. For
            the outside good, gamma is equal to 0.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        :param price: expression for the price, or None if prices are not considered.
        """
        gamma = gamma_parameters[the_id]
        if gamma is None:
            return log(Numeric(1) - alpha_parameters[the_id]) - log(consumption)
        return log(Numeric(1) - alpha_parameters[the_id]) - log(consumption + gamma)

    def calculate_inverse_determinant(
        the_id: int, consumption: Expression
    ) -> Expression:
        """Calculate the inverse of the entries for the determinant. For
            the outside good, gamma is equal to 0.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        :param price: expression for the price, or None if prices are not considered.
        """
        gamma = gamma_parameters[the_id]
        if gamma is None:
            return consumption / (Numeric(1) - alpha_parameters[the_id])

        return (consumption + gamma) / (Numeric(1) - alpha_parameters[the_id])

    utilities = {
        the_id: calculate_utility(the_id, consumption)
        for the_id, consumption in consumed_quantities.items()
    }

    log_determinant_entries = {
        the_id: calculate_log_determinant(the_id, consumption)
        for the_id, consumption in consumed_quantities.items()
    }

    inverse_of_determinant_entries = {
        the_id: calculate_inverse_determinant(the_id, consumption)
        for the_id, consumption in consumed_quantities.items()
    }

    return SpecificModel(
        utilities=utilities,
        log_determinant_entries=log_determinant_entries,
        inverse_of_determinant_entries=inverse_of_determinant_entries,
    )


def mdcev_translated(
    number_of_chosen_alternatives: Expression,
    consumed_quantities: dict[int, Expression],
    baseline_utilities: dict[int, Expression],
    alpha_parameters: dict[int, Expression],
    gamma_parameters: dict[int, Optional[Expression]],
    scale_parameter: Optional[Expression] = None,
):
    """Generate the Biogeme formula for the log probability of the
    MDCEV model using the translated utility function.

    :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`

    :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`

    :param baseline_utilities: see the module documentation :mod:`biogeme.mdcev`

    :param alpha_parameters: see the module documentation :mod:`biogeme.mdcev`

    :param gamma_parameters: see the module documentation :mod:`biogeme.mdcev`

    :param scale_parameter: see the module documentation :mod:`biogeme.mdcev`

    A detailed explanation is provided in the technical report
    "Estimating the MDCEV model with Biogeme"

    """

    specific_model = translated(
        baseline_utilities,
        consumed_quantities,
        alpha_parameters,
        gamma_parameters,
    )

    return mdcev(
        number_of_chosen_alternatives=number_of_chosen_alternatives,
        consumed_quantities=consumed_quantities,
        specific_model=specific_model,
        scale_parameter=scale_parameter,
    )
