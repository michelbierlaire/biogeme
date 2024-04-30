"""Implements the MDCEV model as a generic class

Michel Bierlaire
Sun Apr 7 16:52:33 2024
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from biogeme.biogeme import BIOGEME, Parameters
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Expression, Elem, bioMultSum, log, exp, Beta, Numeric
from biogeme.function_output import FunctionOutput
from biogeme.results import bioResults
from biogeme.tools.checks import validate_dict_types

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
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
        self._estimation_results: bioResults | None = None
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

        validate_dict_types(self.baseline_utilities, 'baseline_utilities', Expression)
        validate_dict_types(self.gamma_parameters, 'gamma_parameters', Expression)
        if self.alpha_parameters is not None:
            validate_dict_types(self.alpha_parameters, 'alpha_parameters', Expression)
        if self.scale_parameter is not None:
            if not isinstance(self.scale_parameter, Expression):
                error_msg = f'Expecting a Biogeme expression and not {type(self.scale_parameter)}'
                raise BiogemeError(error_msg)

    @property
    def estimation_results(self) -> bioResults | None:
        """Property for the estimation results"""
        return self._estimation_results

    @estimation_results.setter
    def estimation_results(self, the_results: bioResults):
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
        assert one_observation.get_sample_size() == 1
        if self.estimation_results:
            return self.baseline_utilities[alternative_id].get_value_c(
                database=one_observation,
                betas=self.estimation_results.get_beta_values(),
                prepare_ids=True,
            )[0]

        return self.baseline_utilities[alternative_id].get_value_c(
            database=one_observation,
            prepare_ids=True,
        )[0]

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

        :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`
        :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`

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
        user_notes: str | None = None,
        biogeme_parameters: str | Parameters | None = None,
    ) -> bioResults:
        """Generate the Biogeme formula for the log probability of the MDCEV model

        :param database: data needed for the estimation of the parameters, in Biogeme format.
        :param number_of_chosen_alternatives: see the module documentation :mod:`biogeme.mdcev`
        :param consumed_quantities: see the module documentation :mod:`biogeme.mdcev`
        :param user_notes: notes to include in Biogeme's estimation report.
        :param biogeme_parameters: parameters controlling the run of Biogeme. The TOML filename or the Parameter
        object should be provided.

        A detailed explanation is provided in the technical report
        "Estimating the MDCEV model with Biogeme":
        """
        logprob = self.loglikelihood(
            number_of_chosen_alternatives=number_of_chosen_alternatives,
            consumed_quantities=consumed_quantities,
        )

        # Create the Biogeme object
        if self.weights is None:
            the_biogeme = BIOGEME(database, logprob)
        else:
            formulas = {'log_like': logprob, 'weight': self.weights}
            the_biogeme = BIOGEME(
                database, formulas, userNotes=user_notes, parameters=biogeme_parameters
            )
        the_biogeme.modelName = self.model_name
        self.estimation_results = the_biogeme.estimate()
        return self.estimation_results

    @abstractmethod
    def _list_of_expressions(self) -> list[Expression]:
        """Extract the list of expressions involved in the model"""
        pass

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
                self.alpha_parameters.values() if self.alpha_parameters else [],
                [self.scale_parameter] if self.scale_parameter else [],
                [self.weights] if self.weights else [],
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
        pass

    @abstractmethod
    def utility_expression_one_alternative(
        self,
        the_id: int,
        the_consumption: Expression,
        unscaled_epsilon: Expression,
    ) -> Expression:
        """Utility expression. Used only for code validation."""
        pass

    def sum_of_utilities(
        self, consumptions: np.ndarray, epsilon: np.ndarray, data_row: Database
    ) -> float:
        """Calculates the sum of all utilities. Used for forecasting."""
        utilities = [
            self.utility_one_alternative(
                the_id=key,
                the_consumption=float(consumptions[index]),
                epsilon=float(epsilon[index]),
                one_observation=data_row,
            )
            for index, key in enumerate(self.index_to_key)
        ]

        return np.sum(utilities)

    def forecast_bruteforce_one_draw(
        self, database: Database, total_budget: float, epsilon: np.ndarray
    ) -> dict[int, float]:
        """Forecast the optimal expenditures given a budget and one realization of epsilon for each alternative"""

        assert len(database.data) == 1
        self.database = database

        def objective_function(x: np.ndarray) -> float:
            result = self.sum_of_utilities(
                consumptions=x, epsilon=epsilon, data_row=database
            )

            return -result

        def budget_constraint(x: np.ndarray) -> float:
            return total_budget - x.sum()

        constraints = {
            'type': 'eq',
            'fun': budget_constraint,
        }

        number_of_variables = len(self.alternatives)
        bounds = [(0, None) for _ in range(number_of_variables)]
        initial_guess = np.array([10] * number_of_variables)

        optimization_result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        optimal_consumption = {
            self.index_to_key[index]: optimization_result.x[index]
            for index in range(number_of_variables)
        }
        for alternative in self.alternatives:
            if alternative not in optimal_consumption:
                optimal_consumption[alternative] = 0
        return optimal_consumption

    @abstractmethod
    def derivative_utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Used in the optimization problem solved for forecasting tp calculate the dual variable."""
        pass

    def optimal_consumption(
        self,
        chosen_alternatives: Iterable[int],
        dual_variable: float,
        epsilon: np.ndarray,
        one_observation: Database,
    ) -> dict[int, float]:
        """Analytical calculation of the optimal consumption if the dual variable is known."""

        result = {
            alt_id: self.optimal_consumption_one_alternative(
                the_id=alt_id,
                dual_variable=dual_variable,
                epsilon=float(epsilon[self.key_to_index[alt_id]]),
                one_observation=one_observation,
            )
            for alt_id in chosen_alternatives
        }
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
        pass

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

    def is_next_alternative_chosen(
        self,
        chosen_alternatives: set[int],
        candidate_alternative: int,
        derivative_zero_expenditure_candidate: float,
        database: Database,
        total_budget: float,
        epsilon: np.ndarray,
    ) -> tuple[bool, float | None]:
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
        }

        # Sort the dictionary by values (descending order)
        ordered_alternatives: list[int] = [
            alt_id
            for alt_id, _ in sorted(
                the_unsorted_w.items(), key=lambda item: item[1], reverse=True
            )
        ]

        chosen_alternatives: set[int] = set()

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
        database: Database,
        total_budget: float,
        epsilon: np.ndarray,
        tolerance_dual=1.0e-4,
        tolerance_budget=1.0e-4,
    ) -> dict[int, float]:

        chosen_alternatives, lower_bound, upper_bound = (
            self.identification_chosen_alternatives(
                database=database, total_budget=total_budget, epsilon=epsilon
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
        for _ in range(1000):
            if not continue_iterations:
                break
            dual_variable = (lower_bound + upper_bound) / 2
            optimal_consumption = self.optimal_consumption(
                chosen_alternatives=chosen_alternatives,
                dual_variable=dual_variable,
                epsilon=epsilon,
                one_observation=database,
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
            one_observation=database,
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
        result: FunctionOutput = utility.get_value_and_derivatives(
            database=one_row, prepare_ids=True, gradient=True, named_results=True
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
        number_of_draws=100,
        brute_force: bool = False,
        tolerance_dual: float = 1.0e-4,
        tolerance_budget: float = 1.0e-4,
        user_defined_epsilon: list[np.ndarray] | None = None,
    ) -> list[pd.DataFrame]:
        """Forecast the optimal expenditures given a budget and one realization of epsilon for each alternative

        :param database: database containing the values of the explanatory variables.
        :param total_budget: total budget.
        :param number_of_draws: number of draws for the error terms.
        :param brute_force: if True, the brute force algorithm is applied. It solves each optimization problem using
            the scipy optimization algorithm. If False, the method proposed by Pinjari and Bhat (2021) is used.
        :param tolerance_dual: convergence criterion for the estimate of the dual variable.
        :param tolerance_budget: convergence criterion for the estimate of the total budget.
        :param user_defined_epsilon: draws from the error terms provided by the user. Each entry of the list corresponds
             to an observation in the database, and consists of a data frame of
             size=(number_of_draws, self.number_of_alternatives). If None, it is generated automatically.
        :return: a list of data frames, each containing the results for each draw

        .. [PinjBhat21] A. R. Pinjari, C. Bhat, Computationally efficient forecasting procedures for Kuhn-Tucker
            consumer demand model systems: Application to residential energy consumption analysis, Journal of Choice
            Modelling, Volume 39, 2021, 100283.
        """

        rows_of_database = [
            Database(name=f'row_{i}', pandas_database=database.data.iloc[[i]])
            for i in range(len(database.data))
        ]
        if len(rows_of_database) == 0:
            error_msg = 'Empty database'
            raise BiogemeError(error_msg)

        if user_defined_epsilon is not None:
            if len(user_defined_epsilon) != len(rows_of_database):
                error_msg = (
                    f'User provided draws have {len(user_defined_epsilon)} entries while there are '
                    f'{len(rows_of_database)} observations in the sample.'
                )
                raise BiogemeError(error_msg)

            for index, epsilon in enumerate(user_defined_epsilon):
                if epsilon.shape[0] != number_of_draws:
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
            epsilons = (
                user_defined_epsilon[index]
                if user_defined_epsilon is not None
                else np.random.gumbel(
                    loc=0, scale=1, size=(number_of_draws, self.number_of_alternatives)
                )
            )

            # Forecasting

            optimal_expenditures = (
                [
                    self.forecast_bruteforce_one_draw(
                        database=row, total_budget=total_budget, epsilon=epsilon
                    )
                    for epsilon in epsilons
                ]
                if brute_force
                else [
                    self.forecast_bisection_one_draw(
                        database=row,
                        total_budget=total_budget,
                        epsilon=epsilon,
                        tolerance_budget=tolerance_budget,
                        tolerance_dual=tolerance_dual,
                    )
                    for epsilon in epsilons
                ]
            )

            gather_data = defaultdict(list)

            # Collecting values for each key
            for optimal_solution in optimal_expenditures:
                for key, value in optimal_solution.items():
                    gather_data[key].append(value)

            all_results.append(pd.DataFrame(gather_data).sort_index(axis='columns'))
        return all_results
