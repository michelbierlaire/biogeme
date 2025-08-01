import unittest

from biogeme.catalog import Controller
from biogeme.exceptions import BiogemeError


class TestController(unittest.TestCase):
    def setUp(self):
        self.controller_name = 'my_controller'
        self.specification_names = ('spec_1', 'spec_2')
        self.the_controller = Controller(
            controller_name=self.controller_name,
            specification_names=self.specification_names,
        )

    def test_constructor(self):
        self.assertEqual(self.the_controller.controller_name, self.controller_name)
        self.assertTupleEqual(
            self.the_controller.specification_names, self.specification_names
        )
        self.assertEqual(self.the_controller.current_index, 0)
        self.assertTupleEqual(
            tuple(self.the_controller.dict_of_index.keys()), self.specification_names
        )
        self.assertListEqual(self.the_controller.controlled_catalogs, [])
        self.assertEqual(self.the_controller.controller_size(), 2)

    def test_all_configurations(self):
        all_configurations = self.the_controller.all_configurations()
        expected_result = {'my_controller:spec_1', 'my_controller:spec_2'}
        self.assertSetEqual(all_configurations, expected_result)

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


if __name__ == '__main__':
    unittest.main()
