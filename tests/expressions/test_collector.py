import unittest

from biogeme.expressions import Beta, Draws, MultipleSum, RandomVariable, Variable
from biogeme.expressions.collectors import (
    collect_init_values,
    list_of_all_betas_in_expression,
    list_of_draws_in_expression,
    list_of_fixed_betas_in_expression,
    list_of_free_betas_in_expression,
    list_of_random_variables_in_expression,
    list_of_variables_in_expression,
)


class TestCollectors(unittest.TestCase):
    def setUp(self):
        # Define simple components used in all tests
        self.var1 = Variable('x1')
        self.var2 = Variable('x2')

        self.beta1 = Beta('b1', 1.0, None, None, 0)  # Free
        self.beta2 = Beta('b2', 2.0, None, None, 1)  # Fixed
        self.beta3 = Beta('b3', 10.0, None, None, 0)  # Free

        self.rv = RandomVariable('omega')
        self.draws = Draws('draw_omega', draw_type='NORMAL')

        # Combine them in a small expression tree
        self.expression = MultipleSum(
            [
                self.var1 * self.beta1,
                self.var2 * self.beta2,
                self.rv,
                self.draws,
                self.beta3**self.beta3,
            ]
        )

    def test_collect_init_values(self):
        result = collect_init_values(self.expression)
        expected_result = {'b1': 1.0, 'b3': 10.0}
        self.assertIsInstance(result, dict)
        self.assertDictEqual(result, expected_result)

    def test_list_of_variables_in_expression(self):
        result = list_of_variables_in_expression(self.expression)
        self.assertIsInstance(result, list)
        names = {v.name for v in result}
        self.assertEqual(names, {'x1', 'x2'})

    def test_list_of_all_betas_in_expression(self):
        result = list_of_all_betas_in_expression(self.expression)
        self.assertIsInstance(result, list)
        names = {b.name for b in result}
        self.assertSetEqual(names, {'b1', 'b2', 'b3'})

    def test_list_of_free_betas_in_expression(self):
        result = list_of_free_betas_in_expression(self.expression)
        self.assertIsInstance(result, list)
        self.assertSetEqual({b.name for b in result}, {'b1', 'b3'})

    def test_list_of_fixed_betas_in_expression(self):
        result = list_of_fixed_betas_in_expression(self.expression)
        self.assertIsInstance(result, list)
        self.assertEqual([b.name for b in result], ['b2'])

    def test_list_of_random_variables_in_expression(self):
        result = list_of_random_variables_in_expression(self.expression)
        self.assertIsInstance(result, list)
        self.assertEqual([v.name for v in result], ['omega'])

    def test_list_of_draws_in_expression(self):
        result = list_of_draws_in_expression(self.expression)
        self.assertIsInstance(result, list)
        self.assertEqual([d.name for d in result], ['draw_omega'])


if __name__ == '__main__':
    unittest.main()
