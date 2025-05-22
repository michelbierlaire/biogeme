"""Tests for the controller class

Michel Bierlaire
Thu Apr 17 2025, 08:34:51
"""

import unittest
from biogeme.catalog import Controller
from biogeme.exceptions import BiogemeError
from biogeme.expressions import SELECTION_SEPARATOR


class DummyCatalog:
    """A minimal dummy catalog with a current_index attribute."""

    def __init__(self):
        self.current_index = -1


class TestController(unittest.TestCase):

    def setUp(self):
        self.names = ['spec1', 'spec2', 'spec3']
        self.controller = Controller('myCtrl', self.names)

    def test_constructor_stores_values(self):
        self.assertEqual(self.controller.controller_name, 'myCtrl')
        self.assertEqual(self.controller.specification_names, tuple(self.names))
        self.assertEqual(self.controller.current_index, 0)
        self.assertEqual(self.controller.dict_of_index['spec2'], 1)

    def test_all_configurations(self):
        configs = self.controller.all_configurations()
        expected = {f'myCtrl{SELECTION_SEPARATOR}{name}' for name in self.names}
        self.assertEqual(configs, expected)

    def test_current_name(self):
        self.controller.set_index(2)
        self.assertEqual(self.controller.current_name(), 'spec3')

    def test_set_name_valid(self):
        self.controller.set_name('spec2')
        self.assertEqual(self.controller.current_index, 1)

    def test_set_name_invalid_raises(self):
        with self.assertRaisesRegex(BiogemeError, 'unknown specification'):
            self.controller.set_name('nonexistent')

    def test_set_index_valid_and_updates_catalogs(self):
        dummy = DummyCatalog()
        self.controller.controlled_catalogs.append(dummy)
        self.controller.set_index(1)
        self.assertEqual(self.controller.current_index, 1)
        self.assertEqual(dummy.current_index, 1)

    def test_set_index_out_of_range_low(self):
        with self.assertRaisesRegex(BiogemeError, 'Wrong index -1'):
            self.controller.set_index(-1)

    def test_set_index_out_of_range_high(self):
        with self.assertRaisesRegex(BiogemeError, 'Wrong index 3'):
            self.controller.set_index(3)

    def test_reset_selection(self):
        self.controller.set_index(2)
        self.controller.reset_selection()
        self.assertEqual(self.controller.current_index, 0)

    def test_modify_controller_step_forward_circular(self):
        self.controller.set_index(2)
        self.controller.modify_controller(1, circular=True)
        self.assertEqual(self.controller.current_index, 0)

    def test_modify_controller_step_backward_circular(self):
        self.controller.set_index(0)
        self.controller.modify_controller(-1, circular=True)
        self.assertEqual(self.controller.current_index, 2)

    def test_modify_controller_step_forward_non_circular(self):
        self.controller.set_index(2)
        mod = self.controller.modify_controller(1, circular=False)
        self.assertEqual(self.controller.current_index, 2)
        self.assertEqual(mod, 0)

    def test_modify_controller_step_backward_non_circular(self):
        self.controller.set_index(1)
        mod = self.controller.modify_controller(-2, circular=False)
        self.assertEqual(self.controller.current_index, 0)
        self.assertEqual(mod, 1)

    def test_eq_and_hash(self):
        c1 = Controller('A', ['x'])
        c2 = Controller('A', ['y'])
        c3 = Controller('B', ['x'])
        self.assertEqual(c1, c2)
        self.assertNotEqual(c1, c3)
        self.assertEqual(hash(c1), hash(c2))
        self.assertNotEqual(hash(c1), hash(c3))

    def test_repr_and_str(self):
        r = repr(self.controller)
        s = str(self.controller)
        self.assertEqual(r, repr('myCtrl'))
        self.assertIn('myCtrl', s)
        self.assertIn('spec1', s)


if __name__ == '__main__':
    unittest.main()
