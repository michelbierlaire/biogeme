import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression, Numeric
from biogeme.mdcev.mdcev import Mdcev


# A concrete subclass of Mdcev for testing purposes
class ConcreteMdcev(Mdcev):
    def transformed_utility(self, the_id, the_consumption, epsilon=None) -> Expression:
        return Numeric(the_id)

    def calculate_log_determinant_one_alternative(
        self, the_id, consumption
    ) -> Expression:
        return Numeric(the_id)

    def calculate_inverse_of_determinant_one_alternative(
        self, the_id, consumption
    ) -> Expression:
        return Numeric(the_id)  # Implement abstract method

    def _list_of_expressions(self) -> list[Expression]:
        """Extract the list of expressions involved in the model"""
        return [Numeric(1), Numeric(2)]

    def utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        return 10

    def utility_expression_one_alternative(
        self,
        the_id: int,
        the_consumption: Expression,
        unscaled_epsilon: Expression,
    ) -> Expression:
        """Utility expression. Used only for code validation."""
        beta = Beta('beta', 10, None, None, 0)
        return beta

    def derivative_utility_one_alternative(
        self,
        the_id: int,
        the_consumption: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Used in the optimization problem solved for forecasting tp calculate the dual variable."""
        return 1

    def optimal_consumption_one_alternative(
        self,
        the_id: int,
        dual_variable: float,
        epsilon: float,
        one_observation: Database,
    ) -> float:
        """Analytical calculation of the optimal consumption if the dual variable is known."""
        return dual_variable

    def lower_bound_dual_variable(
        self,
        chosen_alternatives: set[int],
        one_observation: Database,
        epsilon: np.ndarray,
    ) -> float:
        return 0.0


class TestConcreteMdcev(unittest.TestCase):

    def setUp(self):
        # Set up the common attributes used across many tests
        self.model_name = 'TestModel'
        self.baseline_utilities = {1: Mock(spec=Expression), 2: Mock(spec=Expression)}
        self.gamma_parameters = {1: Mock(spec=Expression), 2: None}
        self.alpha_parameters = {1: Mock(spec=Expression), 2: Mock(spec=Expression)}
        self.scale_parameter = Mock(spec=Expression)
        self.weights = Mock(spec=Expression)

        # Create a concrete instance of Mdcev for testing
        self.instance = ConcreteMdcev(
            model_name=self.model_name,
            baseline_utilities=self.baseline_utilities,
            gamma_parameters=self.gamma_parameters,
            alpha_parameters=self.alpha_parameters,
            scale_parameter=self.scale_parameter,
            weights=self.weights,
        )

    def test_initialization(self):
        # Test correct initialization
        self.assertEqual(self.instance.model_name, self.model_name)
        self.assertEqual(self.instance.baseline_utilities, self.baseline_utilities)
        self.assertEqual(self.instance.gamma_parameters, self.gamma_parameters)
        self.assertEqual(self.instance.alpha_parameters, self.alpha_parameters)
        self.assertEqual(self.instance.scale_parameter, self.scale_parameter)
        self.assertEqual(self.instance.weights, self.weights)

    def test_calculate_utilities(self):
        # Test calculation of utilities based on a mock consumption dict
        consumption = {1: Mock(spec=Expression), 2: Mock(spec=Expression)}
        with patch.object(
            self.instance, 'transformed_utility', return_value=Mock(spec=Expression)
        ) as mocked_transformed_utility:
            result = self.instance.calculate_utilities(consumption)
            self.assertEqual(set(result.keys()), set(consumption.keys()))
            mocked_transformed_utility.assert_called()

    def test_calculate_log_determinant_entries(self):
        # Test calculation of log determinant entries
        consumption = {1: Mock(spec=Expression), 2: Mock(spec=Expression)}
        with patch.object(
            self.instance,
            'calculate_log_determinant_one_alternative',
            return_value=Mock(spec=Expression),
        ) as mocked_method:
            result = self.instance.calculate_log_determinant_entries(consumption)
            self.assertEqual(set(result.keys()), set(consumption.keys()))
            self.assertEqual(mocked_method.call_count, len(consumption))

    def test_error_on_duplicate_keys_in_baseline_utilities(self):
        # Test that duplicate keys in baseline_utilities raise an error
        with self.assertRaises(BiogemeError):
            ConcreteMdcev(
                model_name=self.model_name,
                baseline_utilities={1: Mock(), 1: Mock()},  # Duplicate keys
                gamma_parameters=self.gamma_parameters,
            )

    def test_abstract_method_implementation_error(self):
        # Ensure that instantiation without implementing abstract methods raises TypeError
        with self.assertRaises(TypeError):
            Mdcev()  # Attempt to instantiate the abstract base class without a concrete implementation

    def test_forecast_method(self):
        # Testing the forecast method under controlled conditions
        self.instance.alternatives = {1, 2}  # Assuming two alternatives

        # Assuming two observations
        data = pd.DataFrame({'Column1': [1, 2], 'Column2': [1, 2]})
        database = Database('test', dataframe=data)
        number_of_draws = 10
        epsilons = [
            np.random.gumbel(
                loc=0,
                scale=1,
                size=(number_of_draws, 2),
            )
            for _ in range(2)
        ]

        total_budget = 100

        brute_force = False
        tolerance_dual = 1e-4
        tolerance_budget = 1e-4

        with patch.object(
            self.instance, 'forecast_bisection_one_draw', return_value={1: 50, 2: 50}
        ) as mocked_forecast:
            results = self.instance.forecast(
                database,
                total_budget,
                epsilons,
                brute_force,
                tolerance_dual,
                tolerance_budget,
            )
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            self.assertEqual(len(results[0]), number_of_draws)
            mocked_forecast.assert_called()

    def test_validation_method(self):
        # Testing the validation method
        one_row = Mock(spec=Database)
        with patch.object(
            self.instance, 'validation_one_alternative', return_value=[]
        ) as mocked_validation:
            result = self.instance.validation(one_row)
            self.assertIsInstance(result, list)
            mocked_validation.assert_called()

    def test_error_handling(self):
        # Test error handling in method that requires specific inputs
        database = Database('test', pd.DataFrame({'Column1': [1], 'Column2': [1]}))
        with self.assertRaises(KeyError):
            self.instance.calculate_baseline_utility(
                alternative_id=3, one_observation=database
            )  # Non-existent alternative ID


class TestUpdateParametersInExpressions(unittest.TestCase):
    def setUp(self):

        baseline_utilities = {
            1: Beta('beta1', 0, None, None, 0),
            2: Beta('beta2', 0, None, None, 0),
        }
        gamma_parameters = {
            1: Beta('gamma1', 0, None, None, 0),
            2: None,
        }
        alpha_parameters = {
            1: Beta('alpha1', 0, None, None, 0),
            2: Beta('alpha2', 0, None, None, 0),
        }
        scale_parameter = Beta('scale', 0, None, None, 0)
        weights = Numeric(1)

        self.model = ConcreteMdcev(
            model_name='test',
            baseline_utilities=baseline_utilities,
            gamma_parameters=gamma_parameters,
            alpha_parameters=alpha_parameters,
            scale_parameter=scale_parameter,
            weights=weights,
        )

    def test_update_parameters(self):
        # Mock estimation_results to simulate get_beta_values
        # alpha1 is omitted on purpose.

        estimation_results = {
            'beta1': 1,
            'beta2': 2,
            'gamma1': 3,
            'alpha2': 4,
            'scale': 5,
            'any_parameter': 6,
        }
        self.model._estimation_results = MagicMock()
        self.model._estimation_results.get_beta_values.return_value = estimation_results
        self.model._update_parameters_in_expressions()

        self.assertEqual(
            self.model.baseline_utilities[1].get_value(),
            estimation_results['beta1'],
        )
        self.assertEqual(
            self.model.baseline_utilities[2].get_value(),
            estimation_results['beta2'],
        )
        self.assertEqual(
            self.model.gamma_parameters[1].get_value(),
            estimation_results['gamma1'],
        )
        self.assertEqual(
            self.model.alpha_parameters[1].get_value(),
            0,
        )

        self.assertEqual(
            self.model.alpha_parameters[2].get_value(),
            estimation_results['alpha2'],
        )
        self.assertEqual(
            self.model.scale_parameter.get_value(),
            estimation_results['scale'],
        )


if __name__ == "__main__":
    unittest.main()
