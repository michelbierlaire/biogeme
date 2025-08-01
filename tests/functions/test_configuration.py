'''
Test the configuration module

:author: Michel Bierlaire
:date: Sat Apr  8 17:57:33 2023Fri Mar  3 17:48:36 2023

'''

import unittest

from biogeme.catalog import Configuration, SelectionTuple
from biogeme.exceptions import BiogemeError


class TestConfiguration(unittest.TestCase):
    def test_ctor(self):
        s11 = SelectionTuple(controller='catalog1', selection='item1')
        s12 = SelectionTuple(controller='catalog1', selection='item2')
        s21 = SelectionTuple(controller='catalog2', selection='choice1')
        s22 = SelectionTuple(controller='catalog2', selection='choice2')
        with self.assertRaises(BiogemeError):
            _ = Configuration([s11, s12])
        the_conf = Configuration([s11, s22])
        the_id = the_conf.get_string_id()
        the_correct_id = 'catalog1:item1;catalog2:choice2'
        self.assertEqual(the_id, the_correct_id)

        the_empty_conf = Configuration()
        with self.assertRaises(BiogemeError):
            the_empty_conf.selections = [s11, s12]
        the_empty_conf.selections = [s11, s21]
        the_id = the_empty_conf.get_string_id()
        the_correct_id = 'catalog1:item1;catalog2:choice1'
        self.assertEqual(the_id, the_correct_id)

        the_correct_id = 'catalog1:item1;catalog2:choice1'
        another_conf = Configuration.from_string(the_id)
        the_id = another_conf.get_string_id()
        self.assertEqual(the_id, the_correct_id)

        the_dict = {'catalog1': 'item1', 'catalog2': 'choice2'}
        the_correct_id = 'catalog1:item1;catalog2:choice2'
        yet_another_conf = Configuration.from_dict(the_dict)
        the_id = yet_another_conf.get_string_id()
        self.assertEqual(the_id, the_correct_id)

        conf1 = Configuration([s11])
        conf2 = Configuration([s22])
        combined_conf = Configuration.from_tuple_of_configurations((conf1, conf2))
        the_id = combined_conf.get_string_id()
        the_correct_id = 'catalog1:item1;catalog2:choice2'
        self.assertEqual(the_id, the_correct_id)

    def test_set_of_catalogs(self):
        s11 = SelectionTuple(controller='catalog1', selection='item1')
        s21 = SelectionTuple(controller='catalog2', selection='choice1')
        the_conf = Configuration([s11, s21])
        the_set = the_conf.set_of_controllers()
        the_correct_set = {'catalog1', 'catalog2'}
        self.assertSetEqual(the_set, the_correct_set)

    def test_equal(self):
        s11 = SelectionTuple(controller='catalog1', selection='item1')
        s21 = SelectionTuple(controller='catalog2', selection='choice1')
        s22 = SelectionTuple(controller='catalog2', selection='choice2')
        conf1 = Configuration([s11, s21])
        conf2 = Configuration([s11, s21])
        conf3 = Configuration([s11, s22])
        self.assertEqual(conf1, conf2)
        self.assertNotEqual(conf1, conf3)

    def test_get_selection(self):
        s11 = SelectionTuple(controller='catalog1', selection='item1')
        s21 = SelectionTuple(controller='catalog2', selection='choice1')
        the_conf = Configuration([s11, s21])
        cat1 = the_conf.get_selection('catalog1')
        self.assertEqual(cat1, 'item1')
        cat2 = the_conf.get_selection('catalog2')
        self.assertEqual(cat2, 'choice1')
        cat3 = the_conf.get_selection('catalog3')
        self.assertIsNone(cat3)


if __name__ == '__main__':
    unittest.main()
