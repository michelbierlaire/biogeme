"""Implementation of the "gamma profile" MDCEV model. See the technical report.

Michel Bierlaire
Sun Apr 7 17:16:53 2024
"""

import logging

import numpy as np

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, log, exp, Numeric
from biogeme.mdcev.mdcev import Mdcev
from biogeme.tools.checks import validate_dict_types

logger = logging.getLogger(__name__)


class GammaProfile(Mdcev):
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
        """Calculates the utility for one alternative. This is the term V needed in the loglikelihood function"""

        if the_id not in self.alternatives:
            error_msg = (
                f'Alternative id {the_id} is invalid. Valid ids: {self.alternatives}'
            )
            raise BiogemeError(error_msg)
        gamma: Expression | None = self.gamma_parameters[the_id]
        price: Expression | None = None if self.prices is None else self.prices[the_id]

        baseline_utility = self.baseline_utilities[the_id]

        if gamma is None:
            return baseline_utility - log(the_consumption)

        if price is None:
            return baseline_utility + log(gamma) - log(the_consumption + gamma)

        return baseline_utility + log(gamma) - log(the_consumption + price * gamma)

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
        gamma: Expression | None = self.gamma_parameters.get(the_id)

        if gamma is None:
            return -log(consumption)
        if self.prices is None:
            return -log(consumption + gamma)
        price = self.prices[the_id]
        return -log(consumption + price * gamma)

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
        gamma: Expression | None = self.gamma_parameters[the_id]

        if gamma is None:
            return consumption
        if self.prices is None:
            return consumption + gamma
        price = self.prices[the_id]
        return consumption + price * gamma

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
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return exp(baseline_utility + epsilon) * log(the_consumption / price)
        return (
            exp(baseline_utility + epsilon)
            * gamma
            * log(1 + the_consumption / (price * gamma))
        )

    def utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Utility used in the optimization problem solved for forecasting."""
        baseline_utility = self.calculate_baseline_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        price: float = 1 if self.prices is None else self.prices[the_id].get_value()
        gamma: float = self.gamma_parameters[the_id].get_value()
        if gamma is None:
            return np.exp(baseline_utility + epsilon) * np.log(the_consumption / price)
        return (
            np.exp(baseline_utility + epsilon)
            * gamma
            * np.log(1 + the_consumption / (price * gamma))
        )

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
        price: float = 1 if self.prices is None else self.prices[the_id].get_value()
        gamma: float | None = self.gamma_parameters[the_id].get_value()
        if gamma is None:
            return np.exp(baseline_utility + epsilon) / the_consumption
        return (
            np.exp(baseline_utility + epsilon)
            * gamma
            / (the_consumption + price * gamma)
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
        price: float = 1 if self.prices is None else self.prices[the_id].get_value()
        gamma: float | None = self.gamma_parameters[the_id].get_value()
        if gamma is None:
            return np.exp(baseline_utility + epsilon) / dual_variable
        return (
            np.exp(baseline_utility + epsilon) * gamma / dual_variable - price * gamma
        )

    def sorting_utility_one_alternative(
        self,
        alternative_id: int,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Utility used to sort the alternatives. Used in the forecasting algorithm to identify chosen and non-chpsen
        alternatives"""
        baseline_utility = self.calculate_baseline_utility(
            alternative_id=alternative_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        price: float = (
            1 if self.prices is None else self.prices[alternative_id].get_value()
        )

        if self.prices:
            return baseline_utility + epsilon - np.log(self.prices[alternative_id])
        return baseline_utility + epsilon

    def _list_of_expressions(self) -> list[Expression]:
        """Extract the list of expressions involved in the model"""
        if self.prices is None:
            return []
        return [expression for expression in self.prices.values()]
