"""Implements the MDCEV model as a generic class

Michel Bierlaire
Sun Apr 7 16:52:33 2024
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Elem, Expression, Numeric, bioMultSum, exp, log
from biogeme.function_output import FunctionOutput
from biogeme.jax_calculator import (
    evaluate_expression,
    get_value_and_derivatives,
)
from biogeme.results_processing import EstimationResults
from biogeme.tools.checks import validate_dict_types
from .database_utils import mdcev_row_split
from ..floating_point import SMALL_POSITIVE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MdcevConfiguration:
    """Identify various configurations of the model"""

    gamma_is_none: bool
    price_is_none: bool


class Mdcev(ABC):

    def __init__(
        self,
        model_name: str,
        baseline_utilities: dict[int, Expression],
        gamma_parameters: dict[int, Expression | None],
        alpha_parameters: dict[int, Expression] | None = None,
        scale_parameter: Expression | None = None,
        weights: Expression | None = None,
    ) -> None:
        """ """

        self.model_name: str = model_name
        self.baseline_utilities: dict[int, Expression] = baseline_utilities
        self.gamma_parameters: dict[int, Expression | None] = gamma_parameters
        self.alpha_parameters: dict[int, Expression] | None = alpha_parameters
        self.scale_parameter: Expression | None = scale_parameter
        self.weights: Expression | None = weights
        self._estimation_results: EstimationResults | None = None
        self.database: Database | None = None
        # Check the numbering
        self.alternatives: set[int] = set(self.baseline_utilities)
        # Map the indices with the keys
        self.index_to_key: list[int] = [key for key in self.alternatives]
        self.key_to_index: dict[int, int] = {
            key: index for index, key in enumerate(self.index_to_key)
        }
        if len(self.alternatives) != len(list(self.baseline_utilities)):
            error_msg = (
                f'Some alternatives appear more than once in baseline utilities: '
                f'{list(self.baseline_utilities)}'
            )
            raise BiogemeError(error_msg)
        if error_messages := self._verify_dict_keys_against_set(
            dict_to_check=self.gamma_parameters
        ):
            error_msg = f'Gamma parameters: {", ".join(error_messages)}'
            raise BiogemeError(error_msg)

        if self.alpha_parameters:
            if error_messages := self._verify_dict_keys_against_set(
                dict_to_check=self.alpha_parameters
            ):
                error_msg = f'Alpha parameters: {", ".join(error_messages)}'
                raise BiogemeError(error_msg)

        # Check if there is an outside good. It correspond to a gamma parameter set to None.
        none_keys = [
            key for key, value in self.gamma_parameters.items() if value is None
        ]
        if len(none_keys) > 1:
            error_msg = f'Only one outside good is allowed, not {len(none_keys)}'
            raise BiogemeError(error_msg)
        self.outside_good_key = none_keys[0] if len(none_keys) != 0 else None

        validate_dict_types(self.baseline_utilities, 'baseline_utilities', Expression)
        validate_dict_types(self.gamma_parameters, 'gamma_parameters', Expression)
        if self.alpha_parameters is not None:
            validate_dict_types(self.alpha_parameters, 'alpha_parameters', Expression)
        if self.scale_parameter is not None:
            if not isinstance(self.scale_parameter, Expression):
                error_msg = f'Expecting a Biogeme expression and not {type(self.scale_parameter)}'
                raise BiogemeError(error_msg)

    @property
    def outside_good_index(self) -> int | None:
        """Obtain  the index of the outside good."""
        if self.outside_good_key is None:
            return None
        return self.key_to_index[self.outside_good_key]

    @property
    def estimation_results(self) -> EstimationResults | None:
        """Property for the estimation results"""
        return self._estimation_results

    @estimation_results.setter
    def estimation_results(self, the_results: EstimationResults):
        self._estimation_results = the_results
        self._update_parameters_in_expressions()

    @property
    def number_of_alternatives(self) -> int:
        return len(self.alternatives)

    def _verify_dict_keys_against_set(self, dict_to_check: dict[int, Any]) -> list[str]:
        # Convert dictionary keys to a set
        dict_keys = set(dict_to_check.keys())

        # Find keys in the dictionary that are not in the reference set
        extra_keys = dict_keys - self.alternatives

        # Find keys in the reference set that are not in the dictionary
        missing_keys = self.alternatives - dict_keys

        # Constructing the error message based on the findings
        error_messages = []

        if extra_keys:
            error_messages.append(f'Extra alternatives in dictionary: {extra_keys}.')

        if missing_keys:
            error_messages.append(
                f'Missing alternatives from dictionary: {missing_keys}.'
            )
        return error_messages

    @lru_cache
    def calculate_baseline_utility(
        self, alternative_id: int, one_observation: Database
    ) -> float:
        """As this function may be called many times with the same input in forecasting mode, we use the
        lru_cache decorator."""
        assert one_observation.num_rows() == 1
        if self.estimation_results:
            return evaluate_expression(
                expression=self.baseline_utilities[alternative_id],
                numerically_safe=False,
                database=one_observation,
                betas=self.estimation_results.get_beta_values(),
                aggregation=True,
                use_jit=True,
            )

        return evaluate_expression(
            expression=self.baseline_utilities[alternative_id],
            numerically_safe=False,
            database=one_observation,
            aggregation=True,
            use_jit=True,
        )

    @abstractmethod
    def transformed_utility(
        self,
        the_id: int,
        the_consumption: Expression,
    ) -> Expression:
        """Calculates the transformed utility for one alternative. Used for estimation"""
        pass

    def calculate_utilities(
        self, consumption: dict[int, Expression]
    ) -> dict[int, Expression]:

        return {
            alt_id: self.transformed_utility(alt_id, the_consumption)
            for alt_id, the_consumption in consumption.items()
        }

    @abstractmethod
    def calculate_log_determinant_one_alternative(
        self, the_id: int, consumption: Expression
    ) -> Expression:
        pass

    def calculate_log_determinant_entries(
        self, consumption: dict[int, Expression]
    ) -> dict[int, Expression]:
        return {
            alt_id: self.calculate_log_determinant_one_alternative(
                alt_id, the_consumption
            )
            for alt_id, the_consumption in consumption.items()
        }

    @abstractmethod
    def calculate_inverse_of_determinant_one_alternative(
        self, the_id: int, consumption: Expression
    ) -> Expression:
        pass

    def calculate_inverse_of_determinant_entries(
        self, consumption: dict[int, Expression]
    ) -> dict[int, Expression]:
        return {
            alt_id: self.calculate_inverse_of_determinant_one_alternative(
                alt_id, the_consumption
            )
            for alt_id, the_consumption in consumption.items()
        }

    def info_gamma_parameters(self) -> str:
        """Provides logging information about the outside good"""
        none_count = sum(1 for value in self.gamma_parameters.values() if value is None)
        if none_count == 0:
            report = 'No outside good is included in the model.'
            logger.info(report)
            return report
        if none_count == 1:
            report = 'One outside good is included in the model.'
            logger.info(report)
            return report

        report = (
            'Several outside goods are included in the model. If it is intentional, ignore this warning. '
            'If not, update the definition of the gamma parameters. The outside good, if any, must be associated with '
            'None, and all other alternative with a gamma parameter.'
        )
        logger.warning(report)
        return report

    def loglikelihood(
        self,
        number_of_chosen_alternatives: Expression,
        consumed_quantities: dict[int, Expression],
    ) -> Expression:
        """Generate the Biogeme formula for the log probability of the MDCEV model

        :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev_no_outside_good`
        :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev_no_outside_good`

        A detailed explanation is provided in the technical report
        "Estimating the MDCEV model with Biogeme"

        """
        validate_dict_types(consumed_quantities, 'consumed_quantities', Expression)

        utilities = self.calculate_utilities(consumption=consumed_quantities)
        log_determinant_entries = self.calculate_log_determinant_entries(
            consumption=consumed_quantities
        )
        inverse_of_determinant_entries = self.calculate_inverse_of_determinant_entries(
            consumption=consumed_quantities
        )

        # utility of chosen goods
        utility_terms = [
            Elem({0: 0.0, 1: util}, consumed_quantities[i] > 0)
            for i, util in utilities.items()
        ]
        if self.scale_parameter is None:
            baseline_term = bioMultSum(utility_terms)
        else:
            baseline_term = self.scale_parameter * bioMultSum(utility_terms)

        # Determinant: first term
        first_determinant_terms = [
            Elem({0: 0.0, 1: z}, consumed_quantities[i] > 0)
            for i, z in log_determinant_entries.items()
        ]
        first_determinant = bioMultSum(first_determinant_terms)

        # Determinant: second term
        second_determinant_terms = [
            Elem({0: 0.0, 1: z}, consumed_quantities[i] > 0)
            for i, z in inverse_of_determinant_entries.items()
        ]
        second_determinant = log(bioMultSum(second_determinant_terms))

        # Logsum
        if self.scale_parameter is None:
            terms_for_logsum = [exp(util) for util in utilities.values()]
        else:
            terms_for_logsum = [
                exp(self.scale_parameter * util) for util in utilities.values()
            ]
        logsum_term = number_of_chosen_alternatives * log(bioMultSum(terms_for_logsum))

        log_prob = baseline_term + first_determinant + second_determinant - logsum_term
        # Scale parameter
        if self.scale_parameter is not None:
            log_prob += (number_of_chosen_alternatives - 1) * log(self.scale_parameter)

        return log_prob

    def estimate_parameters(
        self,
        database: Database,
        number_of_chosen_alternatives: Expression,
        consumed_quantities: dict[int, Expression],
        **kwargs,
    ) -> EstimationResults:
        """Generate the Biogeme formula for the log probability of the MDCEV model

        :param database: data needed for the estimation of the parameters, in Biogeme format.
        :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev_no_outside_good`
        :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev_no_outside_good`
        :param **kwargs: additional parameters that are transmitted as such to the constructor of the Biogeme object.


        A detailed explanation is provided in the technical report
        "Estimating the MDCEV model with Biogeme":
        """
        logprob = self.loglikelihood(
            number_of_chosen_alternatives=number_of_chosen_alternatives,
            consumed_quantities=consumed_quantities,
        )

        # Create the Biogeme object
        if self.weights is None:
            the_biogeme = BIOGEME(database, logprob, **kwargs)
        else:
            formulas = {'log_like': logprob, 'weight': self.weights}
            the_biogeme = BIOGEME(database, formulas, **kwargs)
        the_biogeme.modelName = self.model_name
        self.estimation_results = the_biogeme.estimate()
        return self.estimation_results

    @abstractmethod
    def _list_of_expressions(self) -> list[Expression]:
        """Extract the list of expressions involved in the model"""
        raise NotImplementedError

    def _update_parameters_in_expressions(self) -> None:
        """Update the value of the unknown parameters in expression, after estimation"""
        if self._estimation_results is None:
            error_msg = 'No estimation result is available'
            raise BiogemeError(error_msg)

        betas = self._estimation_results.get_beta_values()
        all_expressions = [
            expression
            for group in [
                self.baseline_utilities.values(),
                self.gamma_parameters.values(),
                (
                    self.alpha_parameters.values()
                    if self.alpha_parameters is not None
                    else []
                ),
                [self.scale_parameter] if self.scale_parameter is not None else [],
                [self.weights] if self.weights is not None else [],
            ]
            for expression in group
            if expression is not None
        ]

        # ADd the expressions from the child class
        all_expressions += self._list_of_expressions()

        # Update each expression with the new beta values
        for expression in all_expressions:
            expression.change_init_values(betas)

    @abstractmethod
    def utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Utility needed for forecasting"""
        raise NotImplementedError

    @abstractmethod
    def utility_expression_one_alternative(
        self,
        the_id: int,
        the_consumption: Expression,
        unscaled_epsilon: Expression,
    ) -> Expression:
        """Utility expression. Used only for code validation."""
        raise NotImplementedError

    def sum_of_utilities(
        self, consumptions: np.ndarray, epsilon: np.ndarray, data_row: Database
    ) -> float:
        """Calculates the sum of all utilities. Used for forecasting."""
        try:
            utilities = [
                self.utility_one_alternative(
                    the_id=key,
                    the_consumption=float(consumptions[index]),
                    epsilon=float(epsilon[index]),
                    one_observation=data_row,
                )
                for index, key in enumerate(self.index_to_key)
            ]
        except IndexError as e:
            raise e

        return np.sum(utilities)

    def forecast_bruteforce_one_draw(
        self, one_row_database: Database, total_budget: float, epsilon: np.ndarray
    ) -> dict[int, float] | None:
        """Forecast the optimal expenditures given a budget and one realization of epsilon for each alternative.
        If there is an issue with the optimization algorithm, None is returned.
        """

        if len(epsilon) != self.number_of_alternatives:
            error_msg = f'epsilon must be a vector of size {self.number_of_alternatives}, not {epsilon.shape}'
            raise BiogemeError(error_msg)

        if len(one_row_database.dataframe) != 1:
            error_msg = (
                f'Expecting exactly one row, not {len(one_row_database.dataframe)}'
            )
            raise BiogemeError(error_msg)

        self.database = one_row_database

        # If there is an outside good, we need to impose that the corresponding consumption is not zero.
        # To do that, we impose that it is equal to the exponential of a new variable.

        outside_good = self.outside_good_key is not None

        def objective_function(x: np.ndarray) -> float:
            """Objective function. The number of variables is the number of
            alternatives"""
            result = self.sum_of_utilities(
                consumptions=x, epsilon=epsilon, data_row=one_row_database
            )
            return -result

        def budget_constraint(x: np.ndarray) -> float:
            """Budget constraint when there is no outside good. The number of variables is the number of
            alternatives"""
            return total_budget - x.sum()

        def opposite_budget_constraint(x: np.ndarray) -> float:
            """Budget constraint when there is no outside good. The number of variables is the number of
            alternatives"""
            return x.sum() - total_budget

        constraints = [
            {
                'type': 'ineq',
                'fun': budget_constraint,
            },
            {
                'type': 'ineq',
                'fun': opposite_budget_constraint,
            },
        ]

        number_of_alternatives = len(self.alternatives)

        # Bounds
        bounds: list[tuple[float | None, float | None]] = [
            (0, total_budget) for _ in range(number_of_alternatives)
        ]

        if self.outside_good_index is not None:
            bounds[self.outside_good_index] = (SMALL_POSITIVE, total_budget)

        # Starting point
        # We split the budget equally across alternatives
        initial_consumption = total_budget / number_of_alternatives
        # We split equally across alternatives
        initial_guess = np.array([initial_consumption] * number_of_alternatives)

        # There is a bug in the minimize function. It generates a warning that crashes the script.
        # We are converting this warning into an exception in order to catch it.
        # Convert specific warnings to exceptions
        warnings.filterwarnings(
            'error',
            message='Values in x were outside bounds during a minimize step, clipping to bounds',
        )

        try:
            optimization_result = minimize(
                objective_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )
            optimal_consumption = {
                self.index_to_key[index]: optimization_result.x[index]
                for index in range(number_of_alternatives)
            }
            for alternative in self.alternatives:
                if alternative not in optimal_consumption:
                    optimal_consumption[alternative] = 0
            return optimal_consumption
        except RuntimeWarning as e:
            logger.warning(e)
            return None

    @abstractmethod
    def derivative_utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Used in the optimization problem solved for forecasting tp calculate the dual variable."""
        raise NotImplementedError

    def optimal_consumption(
        self,
        chosen_alternatives: Iterable[int],
        dual_variable: float,
        epsilon: np.ndarray,
        one_observation: Database,
    ) -> dict[int, float]:
        """Analytical calculation of the optimal consumption if the dual variable is known."""
        try:
            result = {
                alt_id: self.optimal_consumption_one_alternative(
                    the_id=alt_id,
                    dual_variable=dual_variable,
                    epsilon=float(epsilon[self.key_to_index[alt_id]]),
                    one_observation=one_observation,
                )
                for alt_id in chosen_alternatives
            }
        except KeyError as e:
            raise e
        return result

    @abstractmethod
    def optimal_consumption_one_alternative(
        self,
        the_id: int,
        dual_variable: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Analytical calculation of the optimal consumption if the dual variable is known."""
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def is_next_alternative_chosen(
        self,
        chosen_alternatives: set[int],
        candidate_alternative: int,
        derivative_zero_expenditure_candidate: float,
        database: Database,
        total_budget: float,
        epsilon: np.ndarray,
    ) -> tuple[bool, float | None]:
        # If the candidate alternative is the outside good, it is in the choice set by definition.
        model_lower_bound = self.lower_bound_dual_variable(
            chosen_alternatives=chosen_alternatives,
            one_observation=database,
            epsilon=epsilon,
        )
        if derivative_zero_expenditure_candidate < model_lower_bound:
            # Lower bound is violated. Alternative not in choice set.
            return False, model_lower_bound

        candidate_set = chosen_alternatives | {candidate_alternative}
        # Derive the optimal consumption
        estimate_optimal_consumption_candidate = self.optimal_consumption(
            chosen_alternatives=candidate_set,
            dual_variable=derivative_zero_expenditure_candidate,
            epsilon=epsilon,
            one_observation=database,
        )
        total_consumption_candidate = sum(
            estimate_optimal_consumption_candidate.values()
        )

        if total_budget <= total_consumption_candidate:
            # Total budget is overpassed. Alternative not in the choice set.
            return False, derivative_zero_expenditure_candidate
        return True, None

    def identification_chosen_alternatives(
        self,
        database: Database,
        total_budget: float,
        epsilon: np.ndarray,
    ) -> tuple[set[int], float, float]:
        """Algorithm identifying the chosen alternatives at the optimal solution

        :param database: one row of the database.
        :param total_budget: total budget for the constraint.
        :param epsilon: vector of draws from the error term.
        :return: the set of chosen alternatives, as well as a lower and an upper bound on the dual variable.
        """
        the_unsorted_w: dict[int, float] = {
            alt_id: self.derivative_utility_one_alternative(
                the_id=alt_id,
                one_observation=database,
                the_consumption=0,
                epsilon=float(epsilon[self.key_to_index[alt_id]]),
            )
            for alt_id in self.alternatives
            if alt_id != self.outside_good_key
        }

        # Sort the dictionary by values (descending order)

        ordered_alternatives: list[int] = [
            alt_id
            for alt_id, _ in sorted(
                the_unsorted_w.items(), key=lambda item: item[1], reverse=True
            )
        ]

        chosen_alternatives: set[int] = (
            set() if self.outside_good_key is None else {self.outside_good_key}
        )

        last_chosen_alternative = None
        for candidate_alternative_id in ordered_alternatives:
            derivative_zero_expenditure_candidate = the_unsorted_w[
                candidate_alternative_id
            ]
            next_chosen, lower_bound = self.is_next_alternative_chosen(
                chosen_alternatives=chosen_alternatives,
                candidate_alternative=candidate_alternative_id,
                derivative_zero_expenditure_candidate=derivative_zero_expenditure_candidate,
                database=database,
                total_budget=total_budget,
                epsilon=epsilon,
            )
            if not next_chosen:
                upper_bound = (
                    np.finfo(np.float64).max
                    if last_chosen_alternative is None
                    else the_unsorted_w[last_chosen_alternative]
                )
                return (
                    chosen_alternatives,
                    lower_bound,
                    upper_bound,
                )

            chosen_alternatives |= {candidate_alternative_id}
            last_chosen_alternative = candidate_alternative_id
        # The full choice set is chosen
        lower_bound = self.lower_bound_dual_variable(
            chosen_alternatives=chosen_alternatives,
            one_observation=database,
            epsilon=epsilon,
        )
        upper_bound = the_unsorted_w[ordered_alternatives[-1]]
        return chosen_alternatives, lower_bound, upper_bound

    def forecast_bisection_one_draw(
        self,
        one_row_of_database: Database,
        total_budget: float,
        epsilon: np.ndarray,
        tolerance_dual=1.0e-13,
        tolerance_budget=1.0e-13,
    ) -> dict[int, float]:
        chosen_alternatives, lower_bound, upper_bound = (
            self.identification_chosen_alternatives(
                database=one_row_of_database, total_budget=total_budget, epsilon=epsilon
            )
        )

        if lower_bound > upper_bound:
            error_msg = (
                f'Lower bound: {lower_bound} larger than upper bound: {upper_bound}. '
                f'Chosen alternatives: {chosen_alternatives}'
            )
            raise BiogemeError(error_msg)

        # Estimate the dual variable by bisection
        total_consumption = 0
        continue_iterations = True
        for _ in range(5000):
            if not continue_iterations:
                break
            dual_variable = (lower_bound + upper_bound) / 2
            optimal_consumption = self.optimal_consumption(
                chosen_alternatives=chosen_alternatives,
                dual_variable=dual_variable,
                epsilon=epsilon,
                one_observation=one_row_of_database,
            )
            negative_consumption = any(
                value < 0 for value in optimal_consumption.values()
            )
            if negative_consumption:
                raise ValueError('Negative consumption')

            if negative_consumption:
                # If any consumption is negative, we update the upper bound
                upper_bound = dual_variable
            else:
                total_consumption = sum(optimal_consumption.values())
                if total_consumption < total_budget:
                    upper_bound = dual_variable

                elif total_consumption > total_budget:
                    lower_bound = dual_variable
                # Stopping criteria
                if (upper_bound - lower_bound) <= tolerance_dual:
                    continue_iterations = False
                elif np.abs(total_consumption - total_budget) <= tolerance_budget:
                    continue_iterations = False

        dual_variable = (lower_bound + upper_bound) / 2
        optimal_consumption = self.optimal_consumption(
            chosen_alternatives=chosen_alternatives,
            dual_variable=dual_variable,
            epsilon=epsilon,
            one_observation=one_row_of_database,
        )
        for alternative in self.alternatives:
            if alternative not in optimal_consumption:
                optimal_consumption[alternative] = 0
        return optimal_consumption

    def validation_one_alternative_one_consumption(
        self, alternative_id: int, value_consumption: float, one_row: Database
    ) -> list[str]:
        """Validation of some of the abstract methods."""
        error_messages = []

        value_epsilon = 0.01
        consumption = Beta('consumption', value_consumption, None, None, 0)
        unscaled_epsilon = Numeric(value_epsilon)
        utility = self.utility_expression_one_alternative(
            the_id=alternative_id,
            the_consumption=consumption,
            unscaled_epsilon=unscaled_epsilon,
        )
        result: FunctionOutput = get_value_and_derivatives(
            expression=utility,
            database=one_row,
            gradient=True,
            named_results=True,
            numerically_safe=False,
            use_jit=True,
        )

        # Validate the utility calculation

        calculated_utility_value = self.utility_one_alternative(
            the_id=alternative_id,
            the_consumption=value_consumption,
            epsilon=value_epsilon,
            one_observation=one_row,
        )

        if not np.isclose(result.function, calculated_utility_value):
            error_msg = f'Inconsistent utility values for alt. {alternative_id}: {result.function} and {calculated_utility_value}'
            error_messages.append(error_msg)

        # Validate the derivative

        calculated_derivative = self.derivative_utility_one_alternative(
            the_id=alternative_id,
            the_consumption=value_consumption,
            epsilon=value_epsilon,
            one_observation=one_row,
        )

        the_gradient = result.gradient['consumption']
        if not np.isclose(the_gradient, calculated_derivative):
            error_msg = f'Inconsistent derivatives for alt. {alternative_id}: {the_gradient} and {calculated_derivative}'
            error_messages.append(error_msg)

        return error_messages

    def validation_one_alternative(
        self, alternative_id: int, one_row: Database
    ) -> list[str]:
        values = [1, 10, 100]
        error_messages = sum(
            (
                self.validation_one_alternative_one_consumption(
                    alternative_id=alternative_id,
                    value_consumption=consumption_value,
                    one_row=one_row,
                )
                for consumption_value in values
            ),
            [],
        )
        # Validate the optimal consumption

        dual_variable = 10
        value_epsilon = 0.01

        optimal_consumption = self.optimal_consumption_one_alternative(
            the_id=alternative_id,
            dual_variable=dual_variable,
            epsilon=value_epsilon,
            one_observation=one_row,
        )

        # If we calculate the derivative for this level of consumption, we should find the dual variable.
        calculated_derivative = self.derivative_utility_one_alternative(
            the_id=alternative_id,
            the_consumption=optimal_consumption,
            epsilon=value_epsilon,
            one_observation=one_row,
        )
        if not np.isclose(calculated_derivative, dual_variable):
            error_msg = (
                f'Inconsistent dual variables for alt. {alternative_id}: '
                f'{calculated_derivative} and {dual_variable}'
            )
            error_messages.append(error_msg)

        return error_messages

    def validation(self, one_row: Database) -> list[str]:
        """Validation of some abstract methods for all alternatives."""
        return sum(
            (
                self.validation_one_alternative(alternative_id=the_id, one_row=one_row)
                for the_id in self.alternatives
            ),
            [],
        )

    def forecast(
        self,
        database: Database,
        total_budget: float,
        epsilons: list[np.ndarray],
        brute_force: bool = False,
        tolerance_dual: float = 1.0e-10,
        tolerance_budget: float = 1.0e-10,
    ) -> list[pd.DataFrame]:
        """Forecast the optimal expenditures given a budget and one realization of epsilon for each alternative

        :param database: database containing the values of the explanatory variables.
        :param total_budget: total budget.
        :param epsilons: draws from the error terms provided by the user. Each entry of the list corresponds
             to an observation in the database, and consists of a data frame of
             size=(number_of_draws, self.number_of_alternatives).
        :param brute_force: if True, the brute force algorithm is applied. It solves each optimization problem using
            the scipy optimization algorithm. If False, the method proposed by Pinjari and Bhat (2021) is used.
        :param tolerance_dual: convergence criterion for the estimate of the dual variable.
        :param tolerance_budget: convergence criterion for the estimate of the total budget.
        :return: a list of data frames, each containing the results for each draw

        .. [PinjBhat21] A. R. Pinjari, C. Bhat, Computationally efficient forecasting procedures for Kuhn-Tucker
            consumer demand model systems: Application to residential energy consumption analysis, Journal of Choice
            Modelling, Volume 39, 2021, 100283.
        """

        rows_of_database = [
            Database(name=f'row_{i}', dataframe=database.dataframe.iloc[[i]])
            for i in range(len(database.dataframe))
        ]
        if len(rows_of_database) == 0:
            error_msg = 'Empty database'
            raise BiogemeError(error_msg)

        if len(epsilons) != len(rows_of_database):
            error_msg = (
                f'User provided draws have {len(epsilons)} entries while there are '
                f'{len(rows_of_database)} observations in the sample.'
            )
            raise BiogemeError(error_msg)

        number_of_draws = None
        for index, epsilon in enumerate(epsilons):
            if number_of_draws is None:
                number_of_draws = epsilon.shape[0]
            elif epsilon.shape[0] != number_of_draws:
                error_msg = (
                    f'User defined draws for obs. {index} contains {epsilon.shape[0]} '
                    f'draws instead of {number_of_draws}'
                )
                raise error_msg
            if epsilon.shape[1] != self.number_of_alternatives:
                error_msg = (
                    f'User defined draws for obs. {index} contains {epsilon.shape[1]} '
                    f'alternatives instead of {self.number_of_alternatives}'
                )
                raise error_msg

        all_results = []
        for index, row in enumerate(rows_of_database):
            logger.info(
                f'Forecasting observation {index} / {len(rows_of_database)} [{number_of_draws} draws]'
            )

            # Forecasting

            optimal_expenditures = (
                [
                    self.forecast_bruteforce_one_draw(
                        one_row_database=row, total_budget=total_budget, epsilon=epsilon
                    )
                    for epsilon in epsilons[index]
                ]
                if brute_force
                else [
                    self.forecast_bisection_one_draw(
                        one_row_of_database=row,
                        total_budget=total_budget,
                        epsilon=epsilon,
                        tolerance_budget=tolerance_budget,
                        tolerance_dual=tolerance_dual,
                    )
                    for epsilon in epsilons[index]
                ]
            )

            gather_data = defaultdict(list)

            # Collecting values for each key
            for optimal_solution in optimal_expenditures:
                if optimal_solution is not None:
                    for key, value in optimal_solution.items():
                        gather_data[key].append(value)

            all_results.append(pd.DataFrame(gather_data).sort_index(axis='columns'))
        return all_results

    def forecast_comparison_one_draw(
        self,
        one_row_of_database: Database,
        total_budget: float,
        epsilon: np.array,
        tolerance_dual: float = 1.0e-10,
        tolerance_budget: float = 1.0e-10,
    ) -> dict[int, float] | None:
        """Compares the results from each of the two forecasting algorithms.
        :param one_row_of_database: list of rows from the database containing the values of the explanatory variables.
        :param total_budget: total budget.
        :param epsilon: draws from the error terms provided by the user. Length: self.number_of_alternatives.
        :param tolerance_dual: convergence criterion for the estimate of the dual variable.
        :param tolerance_budget: convergence criterion for the estimate of the total budget.
        :return: None. Warning are triggered if discrepancies are observed.
        """
        if len(epsilon) != self.number_of_alternatives:
            error_msg = f'There are {self.number_of_alternatives} alternatives and {len(epsilon)} epsilons'
            raise BiogemeError(error_msg)

        logger.info('============ Comparison ===================')
        brute_force: dict[int, float] | None = self.forecast_bruteforce_one_draw(
            one_row_database=one_row_of_database,
            total_budget=total_budget,
            epsilon=epsilon,
        )

        consumption_brute_force = (
            np.array([value for key, value in sorted(brute_force.items())])
            if brute_force is not None
            else None
        )
        obj_brute_force = (
            self.sum_of_utilities(
                consumptions=consumption_brute_force,
                epsilon=epsilon,
                data_row=one_row_of_database,
            )
            if brute_force is not None
            else None
        )
        constraint_brute_force = (
            sum(consumption_brute_force) if brute_force is not None else None
        )
        choice_set_brute_force: set[int] | None = None
        if brute_force is None:
            logger.info('Brute force algorithm failed')
        else:
            choice_set_brute_force = set(
                [key for key, value in brute_force.items() if not np.isclose(value, 0)]
            )
            formatted_brute_force = {
                k: f'{value:.3g}' for k, value in brute_force.items()
            }
            logger.info(
                f'Brute force: {formatted_brute_force} objective {obj_brute_force:.3g}, constraint {constraint_brute_force:.3g}, choice set {choice_set_brute_force}'
            )
        analytical = self.forecast_bisection_one_draw(
            one_row_of_database=one_row_of_database,
            total_budget=total_budget,
            epsilon=epsilon,
            tolerance_budget=tolerance_budget,
            tolerance_dual=tolerance_dual,
        )
        choice_set_analytical: set[int] = set(
            [key for key, value in analytical.items() if not np.isclose(value, 0)]
        )
        consumption_analytical = (
            np.array([value for key, value in sorted(analytical.items())])
            if analytical is not None
            else None
        )
        try:
            obj_analytical = (
                self.sum_of_utilities(
                    consumptions=consumption_analytical,
                    epsilon=epsilon,
                    data_row=one_row_of_database,
                )
                if analytical is not None
                else None
            )
        except RuntimeWarning as e:
            logger.warning(f'{analytical=}')
            logger.warning(f'{consumption_analytical=}')
            raise e

        constraint_analytical = (
            sum(consumption_analytical) if analytical is not None else None
        )
        if analytical is None:
            logger.info('Analytical algorithm failed')
        else:
            formatted_analytical = {
                k: f'{value:.3g}' for k, value in analytical.items()
            }
            logger.info(
                f'Analytical:  {formatted_analytical} objective {obj_analytical:.3g}, constraint {constraint_analytical:.3g}, choice set {choice_set_analytical}'
            )
        if brute_force is None and analytical is None:
            logger.warning('Both algorithms failed.')
            return
        if brute_force is None:
            logger.warning('Brute force algorithm failed.')
            return
        if analytical is None:
            logger.warning('Analytical algorithm failed.')
            return
        if (
            choice_set_brute_force != choice_set_analytical
            and choice_set_brute_force is not None
        ):
            logger.warning(
                f'Different optimal choice sets: analytical {choice_set_analytical}, brute force {choice_set_brute_force}'
            )
        if not np.isclose(obj_analytical, obj_brute_force):
            logger.warning(
                f'Difference between optimal utility with analytical [{obj_analytical:.4g}] and brute '
                f'force [{obj_brute_force:.4g}] algorithms.'
            )
            brute_force_results = ', '.join(
                [f'{key}: {value:.3g}' for key, value in brute_force.items()]
            )
            analytical_results = ', '.join(
                [f'{key}: {value:.3g}' for key, value in analytical.items()]
            )
            logger.warning(f'Solution with brute force: {brute_force_results}')
            logger.warning(f'Solution with analytical: {analytical_results}')
        if not np.isclose(constraint_analytical, constraint_brute_force):
            logger.warning(
                f'Difference between constraint with analytical [{constraint_analytical}] and brute force '
                f'[{constraint_brute_force}] algorithms.'
            )

    def generate_epsilons(
        self, number_of_observations: int, number_of_draws: int
    ) -> list[np.ndarray]:
        """Generate draws for the error terms.

        :param number_of_observations: number of entries in the database.
        :param number_of_draws: number of draws to generate
        :return: a list of length "number_of_observations". Each element is a
        number_of_draws x self.number_of_alternatives array.
        """
        return [
            np.random.gumbel(
                loc=0, scale=1, size=(number_of_draws, self.number_of_alternatives)
            )
            for _ in range(number_of_observations)
        ]

    def validate_forecast(
        self,
        database: Database,
        total_budget: float,
        epsilons: list[np.ndarray],
        tolerance_dual: float = 1.0e-13,
        tolerance_budget: float = 1.0e-13,
    ) -> None:
        """Compare the two algorithms to forecast the optimal expenditures given a budget and one realization of
        epsilon for each alternative

        :param database: database containing the values of the explanatory variables.
        :param total_budget: total budget.
        :param epsilons: draws from the error terms. Each entry of the list corresponds
             to an observation in the database, and consists of a data frame of
             size=(number_of_draws, self.number_of_alternatives).
        :param tolerance_dual: convergence criterion for the estimate of the dual variable.
        :param tolerance_budget: convergence criterion for the estimate of the total budget.

        :return: None. Warning are triggered if discrepancies are observed

        """

        rows_of_database = mdcev_row_split(database)
        if len(rows_of_database) == 0:
            error_msg = 'Empty database'
            raise BiogemeError(error_msg)

        if len(epsilons) != len(rows_of_database):
            error_msg = (
                f'User provided draws have {len(epsilons)} entries while there are '
                f'{len(rows_of_database)} observations in the sample.'
            )
            raise BiogemeError(error_msg)

        number_of_draws = len(epsilons[0])
        for index, row in enumerate(rows_of_database):
            logger.info(
                f'Forecasting observation {index} / {len(rows_of_database)} [{number_of_draws} draws]'
            )

            # Forecasting
            for epsilon in epsilons[index]:
                self.forecast_comparison_one_draw(
                    total_budget=total_budget,
                    one_row_of_database=row,
                    epsilon=epsilon,
                    tolerance_budget=tolerance_budget,
                    tolerance_dual=tolerance_dual,
                )
