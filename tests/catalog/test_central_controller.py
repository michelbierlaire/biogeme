import unittest


from biogeme.catalog import (
    Controller,
    Catalog,
    CentralController,
    Configuration,
    SelectionTuple,
    ControllerOperator,
)
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Numeric, NamedExpression


class TestCentralController(unittest.TestCase):
    def setUp(self):

        expr_1 = Numeric(1)
        expr_2 = Numeric(2)
        expr_3 = Numeric(3)
        expr_4 = Numeric(4)
        expr_5 = Numeric(5)
        expr_6 = Numeric(6)
        expr_7 = Numeric(7)

        # Non nested catalogs
        named_expr_1 = NamedExpression(name='option1', expression=expr_1)
        named_expr_2 = NamedExpression(name='option2', expression=expr_2)
        test_controller_1 = Controller("c1", ["option1", "option2"])
        catalog_1 = Catalog(
            catalog_name='catalog_1',
            named_expressions=[named_expr_1, named_expr_2],
            controlled_by=test_controller_1,
        )
        named_expr_3 = NamedExpression(name='optionA', expression=expr_3)
        named_expr_4 = NamedExpression(name='optionB', expression=expr_4)
        named_expr_5 = NamedExpression(name='optionC', expression=expr_5)
        test_controller_2 = Controller("c2", ["optionA", "optionB", "optionC"])
        catalog_2 = Catalog(
            catalog_name='catalog_2',
            named_expressions=[named_expr_3, named_expr_4, named_expr_5],
            controlled_by=test_controller_2,
        )
        named_expr_6 = NamedExpression(name='choiceX', expression=expr_6)
        named_expr_7 = NamedExpression(name='choiceY', expression=expr_7)
        test_controller_3 = Controller("c3", ["choiceX", "choiceY"])
        catalog_3 = Catalog(
            catalog_name='catalog_3',
            named_expressions=[named_expr_6, named_expr_7],
            controlled_by=test_controller_3,
        )
        test_expression = catalog_1 + catalog_2 + catalog_3
        self.test_central_controller: CentralController = CentralController(
            expression=test_expression
        )

        # Nested catalogs
        named_expr_1 = NamedExpression(name='expr_1', expression=expr_1)
        named_expr_2 = NamedExpression(name='expr_2', expression=expr_2)
        named_expr_3 = NamedExpression(name='expr_3', expression=expr_3)
        named_expr_4 = NamedExpression(name='expr_4', expression=expr_4)
        controller_1 = Controller(
            controller_name='controller_1', specification_names=['expr_1', 'expr_2']
        )
        catalog_1 = Catalog(
            catalog_name='catalog_1',
            named_expressions=[named_expr_1, named_expr_2],
            controlled_by=controller_1,
        )
        named_catalog_1 = NamedExpression('catalog_1', catalog_1)
        controller_2 = Controller(
            controller_name='controller_2',
            specification_names=['expr_3', 'expr_4', 'catalog_1'],
        )
        catalog_2 = Catalog(
            catalog_name='catalog_2',
            named_expressions=[named_expr_3, named_expr_4, named_catalog_1],
            controlled_by=controller_2,
        )
        the_expression = Numeric(1) + catalog_2
        self.central_controller: CentralController = CentralController(
            expression=the_expression
        )
        self.configs = self.central_controller.all_configurations

    def test_number_of_configurations(self):
        # Add test cases for the number of configurations

        self.assertEqual(6, self.central_controller.number_of_configurations())

    def test_get_configuration(self):
        # Add test cases for get_configuration
        # Ensure the returned configuration matches the initial state
        expected_config = Configuration(
            [
                SelectionTuple(controller='controller_1', selection='expr_1'),
                SelectionTuple(controller='controller_2', selection='catalog_1'),
            ]
        )
        self.central_controller.set_configuration(expected_config)
        self.assertEqual(self.central_controller.get_configuration(), expected_config)

    def test_set_configuration(self):
        # Add test cases for set_configuration
        # Ensure it correctly sets the controller state
        config_to_set = Configuration(
            [
                SelectionTuple(controller='controller_1', selection='expr_1'),
                SelectionTuple(controller='controller_2', selection='catalog_1'),
            ]
        )
        self.central_controller.set_configuration(config_to_set)
        self.assertEqual(self.central_controller.get_configuration(), config_to_set)

        # Test for incorrect configuration
        with self.assertRaises(BiogemeError):
            self.central_controller.set_configuration(
                Configuration([SelectionTuple(controller='c4', selection='wrong')])
            )

    def test_prepare_operators(self):
        # Add test cases for prepare_operators
        operators = self.test_central_controller.prepare_operators()

        # Make sure the expected number of operators is generated
        #    Increase: 3
        #    Decrease: 3
        #    Pairs: 3 * 2 * 4 = 24
        #    Several increase: 1
        #    Several increase: 1
        #    Total: 32
        self.assertEqual(len(operators), 32)
        # Test a specific operator function, e.g., 'Increase c1'
        operator: ControllerOperator = operators['Increase c1']
        current_config = Configuration.from_string('c1:option1;c2:optionA;c3:choiceX')
        new_config, steps = operator(current_config=current_config, step=1)
        self.assertEqual(str(new_config), 'c1:option2;c2:optionA;c3:choiceX')
        self.assertEqual(steps, 1)

    def test_two_controllers(self):
        # Add test cases for two_controllers
        # Test for NE direction
        current_config = Configuration.from_string('c1:option1;c2:optionA;c3:choiceX')
        new_config, steps = self.test_central_controller.two_controllers(
            first_controller_name='c1',
            second_controller_name='c2',
            direction='NE',
            current_config=current_config,
            step=1,
        )
        self.assertEqual(str(new_config), 'c1:option2;c2:optionB;c3:choiceX')
        self.assertEqual(steps, 1)

        # Test for invalid direction
        with self.assertRaises(BiogemeError):
            self.central_controller.two_controllers(
                first_controller_name='c1',
                second_controller_name='c2',
                direction='invalid',
                current_config=current_config,
                step=1,
            )

    def test_modify_random_controllers(self):
        # Add test cases for modify_random_controllers
        # Test increasing controllers
        current_config = Configuration.from_string('c1:option1;c2:optionA;c3:choiceX')
        new_config, steps = self.test_central_controller.modify_random_controllers(
            current_config=current_config, increase=True, step=2
        )
        self.assertEqual(steps, 2)

        # Test decreasing controllers
        new_config, steps = self.test_central_controller.modify_random_controllers(
            current_config=current_config, increase=False, step=2
        )
        self.assertEqual(steps, 2)

    def test_set_controller(self):
        # Add test cases for set_controller
        # Test setting c1 to index 1
        self.test_central_controller.set_controller('c1', 1)
        expected_config = Configuration(
            [
                SelectionTuple(controller='c1', selection='option2'),
                SelectionTuple(controller='c2', selection='optionA'),
                SelectionTuple(controller='c3', selection='choiceX'),
            ]
        )
        self.assertEqual(
            self.test_central_controller.get_configuration(), expected_config
        )

        # Test setting an unknown controller
        with self.assertRaises(BiogemeError):
            self.test_central_controller.set_controller('c4', 1)

    def test_decreased_controller(self):
        # Add test cases for decreased_controller
        # Test decreasing c1 by 1
        current_config = Configuration.from_string('c1:option1;c2:optionA;c3:choiceX')
        new_config, steps = self.test_central_controller.decreased_controller(
            current_config=current_config, controller_name='c1', step=1
        )
        self.assertEqual(str(new_config), 'c1:option2;c2:optionA;c3:choiceX')
        self.assertEqual(steps, 1)

        # Test decreasing an unknown controller
        with self.assertRaises(BiogemeError):
            self.test_central_controller.decreased_controller(
                controller_name='c4', current_config=current_config, step=1
            )

    def test_expression_iterator(self):
        for named_expression in self.test_central_controller.expression_iterator():
            self.assertIn(
                named_expression.name,
                self.test_central_controller.all_configurations_ids,
            )

        for named_expression in self.central_controller.expression_iterator():
            self.assertIn(
                named_expression.name,
                self.central_controller.all_configurations_ids,
            )
