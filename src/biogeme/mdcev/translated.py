"""Implementation of the "translated" MDCEV model. See the technical report.

Michel Bierlaire
Tue Apr 9 09:12:55 2024
"""

import numpy as np

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, log, Numeric, exp
from biogeme.mdcev.mdcev import Mdcev


class Translated(Mdcev):
    """Implementation of the "translated" MDCEV model"""

    def __init__(
        self,
        model_name: str,
        baseline_utilities: dict[int, Expression],
        gamma_parameters: dict[int, Expression | None],
        alpha_parameters: dict[int, Expression] | None = None,
        scale_parameter: Expression | None = None,
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

        baseline_utility = self.baseline_utilities[the_id]

        alpha_parameter = self.alpha_parameters[the_id]

        gamma = self.gamma_parameters[the_id]
        if gamma is None:
            return (
                baseline_utility
                + log(alpha_parameter)
                + (alpha_parameter - 1) * log(the_consumption)
            )

        return (
            baseline_utility
            + log(alpha_parameter)
            + (alpha_parameter - 1) * log(the_consumption + gamma)
        )

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
        gamma = self.gamma_parameters[the_id]
        if gamma is None:
            return log(Numeric(1) - self.alpha_parameters[the_id]) - log(consumption)
        return log(Numeric(1) - self.alpha_parameters[the_id]) - log(
            consumption + gamma
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
        gamma = self.gamma_parameters[the_id]
        if gamma is None:
            return consumption / (Numeric(1) - self.alpha_parameters[the_id])
        return (consumption + gamma) / (Numeric(1) - self.alpha_parameters[the_id])

    def utility_expression_one_alternative(
        self,
        the_id: int,
        the_consumption: Expression,
        unscaled_epsilon: Expression,
    ) -> Expression:
        """Utility expression. Used only for code validation."""
        baseline_utility: Expression = self.baseline_utilities[the_id]
        epsilon: Expression = (
            unscaled_epsilon
            if self.scale_parameter is None
            else unscaled_epsilon / self.scale_parameter
        )
        alpha: Expression = self.alpha_parameters[the_id]
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return exp(baseline_utility + epsilon) * the_consumption**alpha
        return exp(baseline_utility + epsilon) * (the_consumption + gamma) ** alpha

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
        alpha = self.alpha_parameters[the_id].get_value()
        gamma = self.gamma_parameters[the_id].get_value()
        if gamma is None:
            return np.exp(baseline_utility + epsilon) * the_consumption**alpha
        return np.exp(baseline_utility + epsilon) * (the_consumption + gamma) ** alpha

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

        alpha = self.alpha_parameters[alternative_id].get_value()
        gamma = self.gamma_parameters[alternative_id].get_value()
        if gamma is None:
            return baseline_utility + epsilon + np.log(alpha)
        return baseline_utility + epsilon + np.log(alpha) + (alpha - 1) * np.log(gamma)

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
        alpha = self.alpha_parameters[the_id].get_value()
        gamma = self.gamma_parameters[the_id].get_value()
        if gamma is None:
            return (
                np.exp(baseline_utility + epsilon)
                * alpha
                * the_consumption ** (alpha - 1.0)
            )
        return (
            np.exp(baseline_utility + epsilon)
            * alpha
            * (the_consumption + gamma) ** (alpha - 1.0)
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
        alpha = self.alpha_parameters[the_id].get_value()
        gamma = self.gamma_parameters[the_id].get_value()
        numerator = dual_variable
        denominator = np.exp(baseline_utility + epsilon) * alpha
        ratio = numerator / denominator
        if gamma is None:
            return ratio ** (1.0 / (alpha - 1))

        return ratio ** (1.0 / (alpha - 1)) - gamma

    def _list_of_expressions(self) -> list[Expression]:
        """Extract the list of expressions involved in the model"""
        return []
