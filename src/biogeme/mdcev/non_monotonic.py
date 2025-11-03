"""Implementation of the "non-monotonic" MDCEV model. See the technical report.

Michel Bierlaire
Tue Apr 9 09:25:20 2024
"""

from functools import lru_cache

import numpy as np

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, exp, log
from biogeme.jax_calculator import evaluate_expression
from biogeme.mdcev.mdcev import Mdcev
from biogeme.tools.checks import validate_dict_types


class NonMonotonic(Mdcev):
    """Implementation of the MDCEV model with non-monotonic utility function."""

    def __init__(
        self,
        model_name: str,
        baseline_utilities: dict[int, Expression],
        gamma_parameters: dict[int, Expression | None],
        mu_utilities: dict[int, Expression],
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
        self.mu_utilities: dict[int, Expression] = mu_utilities
        validate_dict_types(self.mu_utilities, 'mu_utilities', Expression)

    def transformed_utility(
        self,
        the_id: int,
        the_consumption: Expression,
    ) -> Expression:
        """Calculates the utility for one alternative. ."""

        if the_id not in self.alternatives:
            error_msg = (
                f'Alternative id {the_id} is invalid. Valid ids: {self.alternatives}'
            )
            raise BiogemeError(error_msg)
        gamma = self.gamma_parameters[the_id]

        baseline_utility = self.baseline_utilities[the_id]
        mu_utility = self.mu_utilities[the_id]

        alpha_parameter = self.alpha_parameters[the_id]

        if gamma is None:
            return mu_utility + exp(baseline_utility) * the_consumption ** (
                alpha_parameter - 1
            )

        return mu_utility + exp(baseline_utility) * (1 + the_consumption / gamma) ** (
            alpha_parameter - 1
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
            return (
                self.baseline_utilities[the_id]
                + log(1 - self.alpha_parameters[the_id])
                + (self.alpha_parameters[the_id] - 2) * log(consumption)
            )

        return (
            self.baseline_utilities[the_id]
            + log(1 - self.alpha_parameters[the_id])
            - log(gamma)
            + (self.alpha_parameters[the_id] - 2) * log(1 + consumption / gamma)
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
        gamma: Expression = self.gamma_parameters[the_id]

        if gamma is None:
            return (
                exp(-self.baseline_utilities[the_id])
                * consumption ** (2 - self.alpha_parameters[the_id])
                / (1 - self.alpha_parameters[the_id])
            )
        return (
            exp(-self.baseline_utilities[the_id])
            * gamma
            * (1 + consumption / gamma) ** (2 - self.alpha_parameters[the_id])
            / (1 - self.alpha_parameters[the_id])
        )

    @lru_cache
    def calculate_mu_utility(
        self, alternative_id: int, one_observation: Database
    ) -> float:
        """As this function may be called many times with the same input in forecasting mode, we use the
        lru_cache decorator."""
        assert one_observation.num_rows() == 1
        if self.estimation_results:
            return evaluate_expression(
                expression=self.mu_utilities[alternative_id],
                numerically_safe=False,
                database=one_observation,
                betas=self.estimation_results.get_beta_values(),
                aggregation=True,
                use_jit=True,
            )

        return evaluate_expression(
            expression=self.mu_utilities[alternative_id],
            numerically_safe=False,
            database=one_observation,
            aggregation=True,
            use_jit=True,
        )

    def utility_expression_one_alternative(
        self,
        the_id: int,
        the_consumption: Expression,
        unscaled_epsilon: Expression,
    ) -> Expression:
        """Utility expression. Used only for code validation."""
        baseline_utility = self.baseline_utilities[the_id]
        mu_utility = self.mu_utilities[the_id]
        epsilon: Expression = (
            unscaled_epsilon
            if self.scale_parameter is None
            else unscaled_epsilon / self.scale_parameter
        )
        alpha: Expression = self.alpha_parameters[the_id]
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return (
                exp(baseline_utility) * (the_consumption**alpha) / alpha
                + (mu_utility + epsilon) * the_consumption
            )
        return (
            gamma
            * exp(baseline_utility)
            * ((1 + the_consumption / gamma) ** alpha - 1)
            / alpha
            + (mu_utility + epsilon) * the_consumption
        )

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
        mu_utility = self.calculate_mu_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        alpha: float = self.alpha_parameters[the_id].get_value()
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return (
                np.exp(baseline_utility) * (the_consumption**alpha) / alpha
                + (mu_utility + epsilon) * the_consumption
            )
        return (
            gamma.get_value()
            * np.exp(baseline_utility)
            * ((1 + the_consumption / gamma.get_value()) ** alpha - 1)
            / alpha
            + (mu_utility + epsilon) * the_consumption
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
        mu_utility = self.calculate_mu_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        alpha: float = self.alpha_parameters[the_id].get_value()
        gamma: Expression | None = self.gamma_parameters[the_id]
        if gamma is None:
            return (
                np.exp(baseline_utility) * (the_consumption ** (alpha - 1.0))
                + mu_utility
                + epsilon
            )
        return (
            np.exp(baseline_utility)
            * (1.0 + the_consumption / gamma.get_value()) ** (alpha - 1.0)
            + mu_utility
            + epsilon
        )

    @lru_cache
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
        mu_utility = self.calculate_mu_utility(
            alternative_id=the_id, one_observation=one_observation
        )
        if self.scale_parameter is not None:
            epsilon /= self.scale_parameter.get_value()
        alpha: float = self.alpha_parameters[the_id].get_value()
        gamma: Expression | None = self.gamma_parameters[the_id]
        base = (dual_variable - mu_utility - epsilon) * np.exp(-baseline_utility)

        exponent = 1.0 / (alpha - 1.0)
        if gamma is None:
            return base**exponent
        result = gamma.get_value() * (base**exponent - 1)
        return result

    def _list_of_expressions(self) -> list[Expression]:
        """Extract the list of expressions involved in the model"""
        return [expression for expression in self.mu_utilities.values()]

    def lower_bound_dual_variable(
        self,
        chosen_alternatives: set[int],
        one_observation: Database,
        epsilon: np.ndarray,
    ) -> float:
        """Method providing model specific lower bound on the dual variable.

        :param chosen_alternatives: list of alternatives that are chosen at the optimal solution
        :param one_observation: data for one observation.
        :param epsilon: draws from the error term.
        :return: a lower bound and upper bound on the dual variable
        """
        lower_bound = -np.inf
        for alternative_id in chosen_alternatives:
            epsilon_alternative = epsilon[self.key_to_index[alternative_id]]
            if self.scale_parameter is not None:
                epsilon_alternative /= self.scale_parameter.get_value()
            mu_utility = self.calculate_mu_utility(
                alternative_id=alternative_id, one_observation=one_observation
            )
            mu_utility += epsilon_alternative
            if mu_utility > lower_bound:
                lower_bound = mu_utility
        return lower_bound
