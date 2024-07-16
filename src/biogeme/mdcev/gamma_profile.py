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
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None and np.isclose(the_consumption, 0.0):
            error_msg = f'Alternative {the_id} is the outside good. Its consumption cannot be zero.'
            raise BiogemeError(error_msg)
        baseline_utility = self.calculate_baseline_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        price: float = 1 if self.prices is None else self.prices[the_id].get_value()

        if gamma is None:
            return np.exp(baseline_utility + epsilon) * np.log(the_consumption / price)
        return (
            np.exp(baseline_utility + epsilon)
            * gamma.get_value()
            * np.log(1 + the_consumption / (price * gamma.get_value()))
        )

    def derivative_utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Used in the optimization problem solved for forecasting to calculate the dual variable."""

        # For the outside good, the value at zero  consumption is +infinity.
        if the_id == self.outside_good_index and the_consumption == 0.0:
            return np.inf

        baseline_utility = self.calculate_baseline_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        price: float = 1 if self.prices is None else self.prices[the_id].get_value()
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return np.exp(baseline_utility + epsilon) / the_consumption
        return (
            np.exp(baseline_utility + epsilon)
            * gamma.get_value()
            / (the_consumption + price * gamma.get_value())
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
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return np.exp(baseline_utility + epsilon) / dual_variable
        return (
            np.exp(baseline_utility + epsilon) * gamma.get_value() / dual_variable
            - price * gamma.get_value()
        )

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
