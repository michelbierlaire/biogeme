import unittest

import numpy as np
import pandas as pd

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Variable
from biogeme.expressions_registry import ExpressionRegistry


class TestExpressionRegistry(unittest.TestCase):
    def setUp(self):
        # Set up a simple Biogeme model
        self.data = pd.DataFrame(
            {
                'x': [1.0, 2.0, 3.0],
                'y': [4.0, 5.0, 6.0],
            }
        )
        self.database = Database('test', self.data)

        self.beta1 = Beta('beta1', 1.0, -10.0, 10.0, 0)
        self.beta2 = Beta('beta2', 2.0, 0.0, 5.0, 0)
        self.x = Variable('x')
        self.y = Variable('y')

        self.expr = self.beta1 * self.x + self.beta2 * self.y
        self.registry = ExpressionRegistry([self.expr], self.database)

    def test_expressions_property(self):
        self.assertEqual(len(self.registry.expressions), 1)
        self.assertIs(self.registry.expressions[0], self.expr)

    def test_free_betas_extraction(self):
        free_betas = self.registry.free_betas
        free_betas_names = {beta.name for beta in free_betas}
        self.assertIn('beta1', free_betas_names)
        self.assertIn('beta2', free_betas_names)

    def test_fixed_betas_empty(self):
        self.assertEqual(len(self.registry.fixed_betas), 0)

    def test_free_betas_indices(self):
        indices = self.registry.free_betas_indices
        self.assertIn('beta1', indices)
        self.assertIn('beta2', indices)
        self.assertEqual(sorted(indices.values()), [0, 1])

    def test_bounds(self):
        bounds = self.registry.bounds
        self.assertEqual(bounds, [(-10.0, 10.0), (0.0, 5.0)])

    def test_free_betas_values(self):
        values = self.registry.free_betas_init_values
        self.assertEqual(values['beta1'], 1.0)
        self.assertEqual(values['beta2'], 2.0)

    def test_requires_draws_false(self):
        self.assertFalse(self.registry.requires_draws)

    def test_draws_empty(self):
        self.assertEqual(len(self.registry.draws), 0)

    def test_variables_detected(self):
        vars_ = self.registry.variables
        vars_names = [var.name for var in vars_]
        self.assertIn('x', vars_names)
        self.assertIn('y', vars_names)

    def test_variables_indices(self):
        indices = self.registry.variables_indices
        self.assertIn('x', indices)
        self.assertIn('y', indices)

    def test_number_of_free_betas(self):
        self.assertEqual(self.registry.number_of_free_betas, 2)

    def test_draw_types_empty(self):
        self.assertEqual(self.registry.draw_types(), {})

    def test_broadcast_assigns_ids(self):
        # Check that beta1 has a specific_id set
        self.registry.broadcast()
        self.assertIsNotNone(self.registry.free_betas[0].specific_id)
        self.assertIsInstance(self.registry.free_betas[0].specific_id, int)

    def test_get_betas_array_valid_input(self):
        betas_dict = {'beta1': 0.5, 'beta2': 1.5}
        result = self.registry.get_betas_array(betas_dict)
        expected = np.array([0.5, 1.5])
        np.testing.assert_array_equal(result, expected)

    def test_get_betas_array_raises_error(self):
        betas_dict = {'beta1': 0.5, 'unknown_beta': 1.5}
        with self.assertRaises(BiogemeError) as context:
            self.registry.get_betas_array(betas_dict)
        self.assertIn('Unknown parameters', str(context.exception))


if __name__ == '__main__':
    unittest.main()
