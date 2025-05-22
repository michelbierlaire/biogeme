import unittest

from biogeme.catalog import (
    SelectedExpressionsIterator,
    CentralController,
    Configuration,
    Catalog,
    Controller,
)
from biogeme.expressions import Numeric, NamedExpression


class TestSelectedExpressionsIterator(unittest.TestCase):

    def setUp(self):
        # Mock configurations

        expr_1 = Numeric(1)
        expr_2 = Numeric(2)
        expr_3 = Numeric(3)
        named_expr_1 = NamedExpression(name='expr_1', expression=expr_1)
        named_expr_2 = NamedExpression(name='expr_2', expression=expr_2)
        named_expr_3 = NamedExpression(name='expr_3', expression=expr_3)
        controller = Controller(
            controller_name='test', specification_names=['expr_1', 'expr_2', 'expr_3']
        )
        catalog = Catalog(
            catalog_name='test',
            named_expressions=[named_expr_1, named_expr_2, named_expr_3],
            controlled_by=controller,
        )
        self.central_controller = CentralController(expression=catalog)
        self.configs = self.central_controller.all_configurations

    def test_iterator_initialization_with_all_configurations(self):
        iterator = SelectedExpressionsIterator(self.central_controller)
        self.assertEqual(iterator.current_configuration in self.configs, True)

    def test_iterator_initialization_with_selected_configurations(self):
        for selected in self.configs:
            iterator = SelectedExpressionsIterator(self.central_controller, {selected})
            current_configuration = next(iterator)
            self.assertEqual(current_configuration, selected)

    def test_iteration_order(self):
        iterator = SelectedExpressionsIterator(self.central_controller)
        seen = []
        for config in iterator:
            seen.append(config)
        self.assertEqual(set(seen), set(self.configs))
        self.assertEqual(len(seen), len(self.configs))

    def test_iteration_stops(self):
        iterator = SelectedExpressionsIterator(self.central_controller)
        for _ in range(len(self.configs)):
            next(iterator)
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_first_iteration_flag(self):
        iterator = SelectedExpressionsIterator(
            self.central_controller, set(self.configs)
        )
        self.assertTrue(iterator.first)
        _ = next(iterator)
        self.assertFalse(iterator.first)

    def test_number_counter(self):
        iterator = SelectedExpressionsIterator(
            self.central_controller, set(self.configs)
        )
        for i in range(len(self.configs)):
            next(iterator)
            self.assertEqual(iterator.number, i + 1)

    def test_value(self):
        iterator = SelectedExpressionsIterator(
            self.central_controller, set(self.configs)
        )
        expected_values = {'test:expr_2': 2.0, 'test:expr_3': 3.0, 'test:expr_1': 1.0}

        for config in iterator:
            config_id = config.get_string_id()
            the_configuration = Configuration.from_string(config_id)
            self.central_controller.set_configuration(the_configuration)
            value = self.central_controller.expression.get_value()
            self.assertEqual(value, expected_values[config_id])


if __name__ == '__main__':
    unittest.main()
