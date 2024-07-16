"""Implementation of the "generalized translated utility function" MDCEV model. See the technical report.

Michel Bierlaire
Tue Apr 9 08:44:12 2024
"""

from typing import Callable

import numpy as np

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, log, Numeric, exp
from biogeme.mdcev.mdcev import Mdcev, MdcevConfiguration
from biogeme.tools.checks import validate_dict_types


class Generalized(Mdcev):
    """Implementation of the MDCEV model with generalized translated utility function."""

    def __init__(
        self,
        model_name: str,
        baseline_utilities: dict[int, Expression],
        gamma_parameters: dict[int, Expression | None],
        alpha_parameters: dict[int, Expression] | None = None,
        scale_parameter: Expression | None = None,
        prices: dict[int, Expression] | None = None,
        weights: Expression | None = None,
    ) -> None:
        """Implementation of the MDCEV model with Gamma profile."""
        super().__init__(
            model_name=model_name,
            baseline_utilities=baseline_utilities,
            gamma_parameters=gamma_parameters,
            alpha_parameters=alpha_parameters,
            scale_parameter=scale_parameter,
            weights=weights,
        )
        self.prices: dict[int, Expression] | None = prices
        if self.prices is not None:
            validate_dict_types(self.prices, 'prices', Expression)

    def transformed_utility(
        self,
        the_id: int,
        the_consumption: Expression,
    ) -> Expression:
        """Calculates the utility for one alternative."""

        if the_id not in self.alternatives:
            error_msg = (
                f'Alternative id {the_id} is invalid. Valid ids: {self.alternatives}'
            )
            raise BiogemeError(error_msg)
        gamma: Expression | None = self.gamma_parameters[the_id]
        price: Expression | None = None if self.prices is None else self.prices[the_id]

        baseline_utility = self.baseline_utilities[the_id]
        alpha_parameter = self.alpha_parameters[the_id]

        def no_gamma_no_price() -> Expression:
            """gamma is None. price is None"""
            return baseline_utility + (alpha_parameter - 1) * log(the_consumption)

        def no_gamma_yes_price() -> Expression:
            """gamma is None. price is not None"""
            return (
                baseline_utility
                + (alpha_parameter - 1) * log(the_consumption)
                - alpha_parameter * log(price)
            )

        def yes_gamma_no_price() -> Expression:
            """gamma is not None. price is None"""
            return baseline_utility + (alpha_parameter - 1) * log(
                1.0 + the_consumption / gamma
            )

        def yes_gamma_yes_price() -> Expression:
            """gamma is not None. price is not None"""
            return (
                baseline_utility
                - log(price)
                + (alpha_parameter - 1) * log(1.0 + the_consumption / (price * gamma))
            )

        dict_of_configurations: dict[
            MdcevConfiguration, Callable[[], Expression | float]
        ] = {
            MdcevConfiguration(
                gamma_is_none=True, price_is_none=True
            ): no_gamma_no_price,
            MdcevConfiguration(
                gamma_is_none=True, price_is_none=False
            ): no_gamma_yes_price,
            MdcevConfiguration(
                gamma_is_none=False, price_is_none=True
            ): yes_gamma_no_price,
            MdcevConfiguration(
                gamma_is_none=False, price_is_none=False
            ): yes_gamma_yes_price,
        }

        current_configuration = MdcevConfiguration(
            gamma_is_none=gamma is None, price_is_none=price is None
        )

        return dict_of_configurations[current_configuration]()

    def calculate_log_determinant_one_alternative(
        self, the_id: int, consumption: Expression
    ) -> Expression:
        """Calculate the log of the entries for the determinant. For
            the outside good, gamma is equal to 0.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        """
        if the_id not in self.alternatives:
            error_msg = (
                f'Alternative id {the_id} is invalid. Valid ids: {self.alternatives}'
            )
            raise BiogemeError(error_msg)
        price = None if self.prices is None else self.prices[the_id]
        gamma = self.gamma_parameters[the_id]
        if gamma is None:
            return log(Numeric(1) - self.alpha_parameters[the_id]) - log(consumption)
        if price is None:
            return log(Numeric(1) - self.alpha_parameters[the_id]) - log(
                consumption + gamma
            )
        return log(Numeric(1) - self.alpha_parameters[the_id]) - log(
            consumption + price * gamma
        )

    def calculate_inverse_of_determinant_one_alternative(
        self, the_id: int, consumption: Expression
    ) -> Expression:
        """Calculate the inverse of the entries for the determinant. For
            the outside good, gamma is equal to 0.

        :param the_id: identifier of the good.
        :param consumption: expression for the consumption.
        """
        if the_id not in self.alternatives:
            error_msg = (
                f'Alternative id {the_id} is invalid. Valid ids: {self.alternatives}'
            )
            raise BiogemeError(error_msg)
        price = None if self.prices is None else self.prices[the_id]
        gamma = self.gamma_parameters[the_id]
        if gamma is None:
            return consumption / (Numeric(1) - self.alpha_parameters[the_id])
        if price is None:
            return (consumption + gamma) / (Numeric(1) - self.alpha_parameters[the_id])
        return (consumption + price * gamma) / (
            Numeric(1) - self.alpha_parameters[the_id]
        )

    def utility_expression_one_alternative(
        self,
        the_id: int,
        the_consumption: Expression,
        unscaled_epsilon: Expression,
    ) -> Expression:
        """Utility expression. Used only for code validation."""
        baseline_utility = self.baseline_utilities[the_id]
        epsilon: Expression = (
            unscaled_epsilon
            if self.scale_parameter is None
            else unscaled_epsilon / self.scale_parameter
        )
        price: Expression = Numeric(1) if self.prices is None else self.prices[the_id]
        alpha: Expression = self.alpha_parameters[the_id]
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return (
                exp(baseline_utility + epsilon)
                * (the_consumption / price) ** alpha
                / alpha
            )
        the_term = (1 + the_consumption / (price * gamma)) ** alpha
        return exp(baseline_utility + epsilon) * gamma * (the_term - 1) / alpha

    def utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Utility needed for forecasting"""
        baseline_utility = self.calculate_baseline_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        price: float = 1 if self.prices is None else self.prices[the_id].get_value()
        alpha: float = self.alpha_parameters[the_id].get_value()
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return (
                np.exp(baseline_utility + epsilon)
                * (the_consumption / price) ** alpha
                / alpha
            )
        the_term = (1 + the_consumption / (price * gamma.get_value())) ** alpha
        return (
            np.exp(baseline_utility + epsilon)
            * gamma.get_value()
            * (the_term - 1)
            / alpha
        )

    def sorting_utility_one_alternative(
        self,
        alternative_id: int,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Utility used to sort the alternatives. Used in the forecasting algorithm to identify chosen and non chpsen
        alternatives"""
        baseline_utility = self.calculate_baseline_utility(
            alternative_id=alternative_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()

        if self.prices:
            price = self.prices[alternative_id].get_value()
            return baseline_utility + epsilon - np.log(price)
        return baseline_utility + epsilon

    def derivative_utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Used in the optimization problem solved for forecasting tp calculate the dual variable."""
        baseline_utility = self.calculate_baseline_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        price: float = 1.0 if self.prices is None else self.prices[the_id].get_value()
        alpha: float = self.alpha_parameters[the_id].get_value()
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return (
                np.exp(baseline_utility + epsilon)
                * (the_consumption / price) ** (alpha - 1)
                / price
            )
        return (
            np.exp(baseline_utility + epsilon)
            * (1 + the_consumption / (price * gamma.get_value())) ** (alpha - 1)
            / price
        )

    def optimal_consumption_one_alternative(
        self,
        the_id: int,
        dual_variable: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Analytical calculation of the optimal consumption if the dual variable is known."""
        baseline_utility = self.calculate_baseline_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        price: float = 1.0 if self.prices is None else self.prices[the_id].get_value()
        alpha: float = self.alpha_parameters[the_id].get_value()
        gamma: Expression | None = self.gamma_parameters[the_id]
        numerator = price * dual_variable
        denominator = np.exp(baseline_utility + epsilon)
        ratio = numerator / denominator
        exponent = 1 / (alpha - 1)
        if gamma is None:
            return price * ratio**exponent
        return price * gamma.get_value() * (ratio**exponent - 1)

    def lower_bound_dual_variable(
        self,
        chosen_alternatives: set[int],
        one_observation: Database,
        epsilon: np.ndarray,
    ) -> float:
        """Method providing model specific bounds on the dual variable. It not overloaded,
        default values are used.

        :param chosen_alternatives: list of alternatives that are chosen at the optimal solution
        :param one_observation: data for one observation.
        :param epsilon: draws from the error term.
        :return: a lower bound on the dual variable, such that the expenditure calculated for any larger value is
        well-defined and non negative.
        """
        return 0.0

    def _list_of_expressions(self) -> list[Expression]:
        """Extract the list of expressions involved in the model"""
        if self.prices is None:
            return []
        return [expression for expression in self.prices.values()]
