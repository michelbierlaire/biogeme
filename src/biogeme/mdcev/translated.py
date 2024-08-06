"""Implementation of the "translated" MDCEV model. See the technical report.

Michel Bierlaire
Tue Apr 9 09:12:55 2024
"""

import logging

import numpy as np

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, log, Numeric, exp
from biogeme.mdcev.mdcev import Mdcev

logger = logging.getLogger(__name__)

MAX_EXP_ARGUMENT = np.log(np.finfo(dtype=float).max)


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

        gamma: Expression | None = self.gamma_parameters[the_id]
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
        gamma: Expression | None = self.gamma_parameters[the_id]
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
        gamma: Expression | None = self.gamma_parameters[the_id]
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
            log_result = baseline_utility + epsilon + alpha * log(the_consumption)
            return exp(log_result)
        log_result = baseline_utility + epsilon + alpha * log(the_consumption + gamma)
        return exp(log_result)

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
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            if the_consumption == 0.0:
                return 0.0
            log_result = baseline_utility + epsilon + alpha * np.log(the_consumption)
            return np.exp(log_result)
        log_result = (
            baseline_utility
            + epsilon
            + alpha * np.log(the_consumption + gamma.get_value())
        )
        return np.exp(log_result)

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
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            log_result = (
                (
                    baseline_utility
                    + epsilon
                    + np.log(alpha)
                    + (alpha - 1) * np.log(the_consumption)
                )
                if the_consumption != 0.0
                else 0.0
            )
            return np.exp(log_result)
        log_result = (
            baseline_utility
            + epsilon
            + np.log(alpha)
            + (alpha - 1) * np.log(the_consumption + gamma.get_value())
        )
        return np.exp(log_result)

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
        if np.isclose(alpha, 0) or np.isclose(alpha, 1.0):
            error_msg = f'Parameter alpha[{the_id}] must be strictly below 0 and 1: alpha[{the_id}={alpha:.3g}'
            raise BiogemeError(error_msg)
        gamma: Expression | None = self.gamma_parameters[the_id]
        if dual_variable == 0.0:
            return 0.0
        log_ratio = np.log(dual_variable) - baseline_utility - epsilon - np.log(alpha)
        log_result = min(log_ratio / (alpha - 1), MAX_EXP_ARGUMENT)
        if gamma is None:
            return np.exp(log_result)
        return np.exp(log_result) - gamma.get_value()

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
        return []
