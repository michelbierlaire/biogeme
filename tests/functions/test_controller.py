import unittest
from unittest.mock import MagicMock
from biogeme.controller import Controller, CentralController
from biogeme.exceptions import BiogemeError
from biogeme.configuration import Configuration, SelectionTuple


class TestController(unittest.TestCase):
    def setUp(self):
        self.controller_name = 'my_controller'
        self.specification_names = ['spec_1', 'spec_2']
        self.the_controller = Controller(
            controller_name=self.controller_name,
            specification_names=self.specification_names,
        )

    def test_constructor(self):
        self.assertEqual(self.the_controller.controller_name, self.controller_name)
        self.assertListEqual(
            self.the_controller.specification_names, self.specification_names
        )
        self.assertEqual(self.the_controller.current_index, 0)
        self.assertListEqual(
            list(self.the_controller.dict_of_index.keys()), self.specification_names
        )
        self.assertListEqual(self.the_controller.controlled_catalogs, [])
        self.assertEqual(self.the_controller.controller_size(), 2)

    def test_all_configurations(self):
        all_configurations = self.the_controller.all_configurations()
        expected_result = ['my_controller:spec_1', 'my_controller:spec_2']
        self.assertListEqual(all_configurations, expected_result)

    def test_name(self):
        name = 'spec_2'
        self.the_controller.set_name(name)
        current_name = self.the_controller.current_name()
        self.assertEqual(name, current_name)

        incorrect_name = 'incorrect_name'
        with self.assertRaises(BiogemeError):
            self.the_controller.set_name(incorrect_name)

    def test_index(self):
        self.the_controller.set_index(1)
        current_name = self.the_controller.current_name()
        expected_name = 'spec_2'
        self.assertEqual(expected_name, current_name)
        self.the_controller.reset_selection()
        current_name = self.the_controller.current_name()
        expected_name = 'spec_1'
        self.assertEqual(expected_name, current_name)

    def test_modify_controller(self):
        self.the_controller.reset_selection()
        self.the_controller.modify_controller(step=1, circular=False)
        self.assertEqual(self.the_controller.current_index, 1)
        self.the_controller.modify_controller(step=1, circular=False)
        self.assertEqual(self.the_controller.current_index, 1)
        self.the_controller.modify_controller(step=1, circular=True)
        self.assertEqual(self.the_controller.current_index, 0)
        self.the_controller.modify_controller(step=10, circular=False)
        self.assertEqual(self.the_controller.current_index, 1)
        self.the_controller.modify_controller(step=-10, circular=False)
        self.assertEqual(self.the_controller.current_index, 0)


class TestCentralController(unittest.TestCase):
    def setUp(self):
        expression_mock = MagicMock()
        expression_mock.get_all_controllers.return_value = {
            Controller("c1", ["option1", "option2"]),
            Controller("c2", ["optionA", "optionB", "optionC"]),
            Controller("c3", ["choiceX", "choiceY"]),
        }
        self.controllers = expression_mock.get_all_controllers()
        self.central_controller = CentralController(expression_mock)

    def test_number_of_configurations(self):
        # Add test cases for the number of configurations
        self.assertEqual(self.central_controller.number_of_configurations(), 12)

    def test_get_configuration(self):
        # Add test cases for get_configuration
        # Ensure the returned configuration matches the initial state
        expected_config = Configuration(
            [
                SelectionTuple(controller='c1', selection='option1'),
                SelectionTuple(controller='c2', selection='optionA'),
                SelectionTuple(controller='c3', selection='choiceX'),
            ]
        )
        self.central_controller.set_configuration(expected_config)
        self.assertEqual(self.central_controller.get_configuration(), expected_config)

    def test_set_configuration(self):
        # Add test cases for set_configuration
        # Ensure it correctly sets the controller state
        config_to_set = Configuration(
            [
                SelectionTuple(controller='c1', selection='option2'),
                SelectionTuple(controller='c2', selection='optionC'),
                SelectionTuple(controller='c3', selection='choiceY'),
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
        operators = self.central_controller.prepare_operators()

        # Make sure the expected number of operators is generated
        #    Increase: 3
        #    Decrease: 3
        #    Pairs: 3 * 2 * 4 = 24
        #    Several increase: 1
        #    Several increase: 1
        #    Total: 32
        self.assertEqual(len(operators), 32)
        # Test a specific operator function, e.g., 'Increase c1'
        operator = operators['Increase c1']
        current_config = 'c1:option1;c2:optionA;c3:choiceX'
        new_config, steps = operator(current_config=current_config, step=1)
        self.assertEqual(str(new_config), 'c1:option2;c2:optionA;c3:choiceX')
        self.assertEqual(steps, 1)

    def test_two_controllers(self):
        # Add test cases for two_controllers
        # Test for NE direction
        current_config = 'c1:option1;c2:optionA;c3:choiceX'
        new_config, steps = self.central_controller.two_controllers(
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
                current_config, 'c1', 'c2', 1, 'invalid'
            )

    def test_modify_random_controllers(self):
        # Add test cases for modify_random_controllers
        # Test increasing controllers
        current_config = 'c1:option1;c2:optionA;c3:choiceX'
        new_config, steps = self.central_controller.modify_random_controllers(
            current_config=current_config, increase=True, step=2
        )
        self.assertEqual(steps, 2)

        # Test decreasing controllers
        new_config, steps = self.central_controller.modify_random_controllers(
            current_config=current_config, increase=False, step=2
        )
        self.assertEqual(steps, 2)

    def test_set_controller(self):
        # Add test cases for set_controller
        # Test setting c1 to index 1
        self.central_controller.set_controller('c1', 1)
        expected_config = Configuration(
            [
                SelectionTuple(controller='c1', selection='option2'),
                SelectionTuple(controller='c2', selection='optionA'),
                SelectionTuple(controller='c3', selection='choiceX'),
            ]
        )
        self.assertEqual(self.central_controller.get_configuration(), expected_config)

        # Test setting an unknown controller
        with self.assertRaises(BiogemeError):
            self.central_controller.set_controller('c4', 1)

    def test_decreased_controller(self):
        # Add test cases for decreased_controller
        # Test decreasing c1 by 1
        current_config = 'c1:option1;c2:optionA;c3:choiceX'
        new_config, steps = self.central_controller.decreased_controller(
            current_config=current_config, controller_name='c1', step=1
        )
        self.assertEqual(str(new_config), 'c1:option2;c2:optionA;c3:choiceX')
        self.assertEqual(steps, 1)

        # Test decreasing an unknown controller
        with self.assertRaises(BiogemeError):
            self.central_controller.decreased_controller(current_config, 'c4', 1)


if __name__ == '__main__':
    unittest.main()
