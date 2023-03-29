'''
Test the multiple_expressions module

:author: Michel Bierlaire
:date: Fri Mar  3 17:48:36 2023

'''
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest
import biogeme.exceptions as excep
import biogeme.expressions as ex
import biogeme.multiple_expressions as me


class TestMultipleExpressions(unittest.TestCase):
    def setUp(self):
        betas = [ex.Beta(f'beta{i}', i * 0.1, None, None, 0) for i in range(10)]
        g1_first = me.NamedExpression(name='g1_first', expression=betas[0])
        g1_second = me.NamedExpression(name='g1_second', expression=betas[1])
        g1_tuple = (g1_first, g1_second)
        self.catalog_1 = me.Catalog('catalog_1', g1_tuple)
        self.assertEqual(self.catalog_1.current_index, 0)

        g2_zero = me.NamedExpression(name='one', expression=ex.Numeric(1))
        g2_first = me.NamedExpression(name='g2_first', expression=betas[2])
        g2_second = me.NamedExpression(name='g2_second', expression=betas[3])
        g2_third = me.NamedExpression(name='g2_third', expression=betas[4])
        g2_fourth = me.NamedExpression(name='g2_fourth', expression=self.catalog_1)
        g2_tuple = (g2_first, g2_second, g2_third)
        self.catalog_2 = me.Catalog('catalog_2', g2_tuple)
        self.assertEqual(self.catalog_2.current_index, 0)

        g3_tuple = (g2_zero, g2_first, g2_second, g2_third, g2_fourth)
        self.catalog_3 = me.Catalog('catalog_3', g3_tuple)

        one = me.NamedExpression(name='one', expression=ex.Numeric(1))
        two = me.NamedExpression(name='two', expression=ex.Numeric(2))
        three = me.NamedExpression(name='three', expression=ex.Numeric(3))
        four = me.NamedExpression(name='four', expression=ex.Numeric(4))
        five = me.NamedExpression(name='five', expression=ex.Numeric(5))
        six = me.NamedExpression(name='six', expression=ex.Numeric(6))

        c6_tuple = (two, three)
        catalog_6 = me.Catalog('catalog_6', c6_tuple)
        c6 = me.NamedExpression(name='c6', expression=catalog_6)
        c6_tuple = (five, six)
        catalog_7 = me.Catalog('catalog_7', c6_tuple)
        c7 = me.NamedExpression(name='c7', expression=catalog_7)
        c4_tuple = (one, c6)
        catalog_4 = me.Catalog('catalog_4', c4_tuple)
        c5_tuple = (four, c7)
        catalog_5 = me.Catalog('catalog_5', c5_tuple)
        self.complex_expression = catalog_4 + catalog_5

        # Same expression as above, where catalogs are synchronized
        c60_tuple = (two, three)
        catalog_60 = me.Catalog('catalog_60', c60_tuple)
        c60 = me.NamedExpression(name='c60', expression=catalog_60)
        c60_tuple = (five, six)
        catalog_70 = me.SynchronizedCatalog('catalog_70', c60_tuple, catalog_60)
        catalog_71 = me.Catalog('catalog_71', c60_tuple)
        c70 = me.NamedExpression(name='c70', expression=catalog_70)
        c71 = me.NamedExpression(name='c71', expression=catalog_71)
        c40_tuple = (one, c60)
        catalog_40 = me.Catalog('catalog_40', c40_tuple)
        c50_tuple = (four, c70)
        c51_tuple = (four, c71)
        catalog_50 = me.SynchronizedCatalog('catalog_50', c50_tuple, catalog_40)
        catalog_51 = me.SynchronizedCatalog('catalog_51', c51_tuple, catalog_40)
        self.synchronized_expression = catalog_40 + catalog_50
        self.synchronized_expression_2 = catalog_40 + catalog_51

        self.wrong_expression = catalog_6 + catalog_6

    def test_ctor(self):
        empty_tuple = tuple()
        with self.assertRaises(excep.biogemeError):
            _ = me.Catalog('the_name', empty_tuple)

        incorrect = me.NamedExpression(name='incorrect', expression='not_an_expression')
        wrong_tuple = (incorrect,)
        with self.assertRaises(excep.biogemeError):
            _ = me.Catalog('the_name', wrong_tuple)

        duplicate = self.catalog_1 * self.catalog_1
        with self.assertRaises(excep.biogemeError):
            _ = duplicate.dict_of_catalogs()

    def test_wrong_expression(self):
        with self.assertRaises(excep.biogemeError):
            _ = self.wrong_expression.dict_of_catalogs()

    def test_names(self):
        result = [e.name for e in self.catalog_1.tuple_of_named_expressions]
        correct_result = ['g1_first', 'g1_second']
        self.assertListEqual(result, correct_result)

    def test_set_index(self):
        with self.assertRaises(excep.biogemeError):
            self.catalog_1.set_index(99999)

        self.catalog_1.set_index(1)
        self.assertEqual(self.catalog_1.current_index, 1)

    def test_catalog_size(self):
        size = self.catalog_1.catalog_size()
        self.assertEqual(size, 2)

    def test_selected(self):
        self.catalog_1.set_index(1)
        name = self.catalog_1.selected_name()
        correct_name = 'g1_second'
        self.assertEqual(name, correct_name)
        expression = self.catalog_1.selected_expression()
        correct_expression = 'beta1(init=0.1)'
        self.assertEqual(str(expression), correct_expression)

    def test_dict_of_catalogs(self):
        expression = self.catalog_1 + self.catalog_2
        the_dict = expression.dict_of_catalogs()
        correct_keys = ['catalog_1', 'catalog_2']
        self.assertListEqual(list(the_dict.keys()), correct_keys)

        another_expression = ex.Numeric(2) + self.catalog_3
        another_dict = another_expression.dict_of_catalogs()
        correct_keys = ['catalog_1', 'catalog_3']
        self.assertListEqual(list(another_dict.keys()), correct_keys)

    def number_of_multiple_expressions(self):
        expression = self.catalog_1 + self.catalog_2
        the_number = expression.number_of_multiple_expressions()
        correct_number = 6
        self.assertEqual(the_number, correct_number)

        another_expression = ex.Numeric(2) + self.catalog_3
        the_number = another_expression.number_of_multiple_expressions()
        correct_number = 7
        self.assertEqual(the_number, correct_number)

    def test_increment_and_reset(self):
        expression = self.catalog_1 + self.catalog_2

        correct_expression = (
            '([catalog_1: g1_first]beta0(init=0.0) + '
            '[catalog_2: g2_first]beta2(init=0.2))'
        )
        self.assertEqual(str(expression), correct_expression)
        done = expression.increment_selection()
        self.assertEqual(done, True)
        correct_expression = (
            '([catalog_1: g1_first]beta0(init=0.0) + '
            '[catalog_2: g2_second]beta3(init=0.30000000000000004))'
        )

        self.assertEqual(str(expression), correct_expression)

    def test_select_expression(self):
        expression = self.catalog_1 + self.catalog_2
        expression.reset_expression_selection()
        total = expression.select_expression('catalog_2', 1)
        self.assertEqual(total, 1)
        self.assertEqual(self.catalog_1.current_index, 0)
        self.assertEqual(self.catalog_2.current_index, 1)
        total = expression.select_expression('catalog_1', 1)
        self.assertEqual(total, 1)
        self.assertEqual(self.catalog_1.current_index, 1)
        self.assertEqual(self.catalog_2.current_index, 1)

    def test_iterator(self):
        expression = self.catalog_1 + self.catalog_2
        configurations = [configuration for configuration, e in expression]
        correct_configurations = [
            {'catalog_1': 'g1_first', 'catalog_2': 'g2_first'},
            {'catalog_1': 'g1_first', 'catalog_2': 'g2_second'},
            {'catalog_1': 'g1_first', 'catalog_2': 'g2_third'},
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_first'},
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_second'},
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_third'},
        ]
        for the_dict, the_correct_dict in zip(configurations, correct_configurations):
            self.assertDictEqual(the_dict, the_correct_dict)

        size = expression.number_of_multiple_expressions()
        self.assertEqual(size, len(configurations))

        another_expression = ex.Numeric(2) + self.catalog_3
        configurations = [
            configuration for configuration, e in another_expression
        ]
        correct_configurations = [
            {'catalog_3': 'one'},
            {'catalog_3': 'g2_first'},
            {'catalog_3': 'g2_second'},
            {'catalog_3': 'g2_third'},
            {'catalog_1': 'g1_first', 'catalog_3': 'g2_fourth'},
            {'catalog_1': 'g1_second', 'catalog_3': 'g2_fourth'},
        ]
        for the_dict, the_correct_dict in zip(configurations, correct_configurations):
            self.assertDictEqual(the_dict, the_correct_dict)

        size = another_expression.number_of_multiple_expressions()
        self.assertEqual(size, len(configurations))

        configurations = [
            configuration for configuration, e in self.complex_expression
        ]
        correct_configurations = [
            {'catalog_4': 'one', 'catalog_5': 'four'},
            {'catalog_4': 'one', 'catalog_7': 'five', 'catalog_5': 'c7'},
            {'catalog_4': 'one', 'catalog_7': 'six', 'catalog_5': 'c7'},
            {'catalog_6': 'two', 'catalog_4': 'c6', 'catalog_5': 'four'},
            {
                'catalog_6': 'two',
                'catalog_4': 'c6',
                'catalog_7': 'five',
                'catalog_5': 'c7',
            },
            {
                'catalog_6': 'two',
                'catalog_4': 'c6',
                'catalog_7': 'six',
                'catalog_5': 'c7',
            },
            {'catalog_6': 'three', 'catalog_4': 'c6', 'catalog_5': 'four'},
            {
                'catalog_6': 'three',
                'catalog_4': 'c6',
                'catalog_7': 'five',
                'catalog_5': 'c7',
            },
            {
                'catalog_6': 'three',
                'catalog_4': 'c6',
                'catalog_7': 'six',
                'catalog_5': 'c7',
            },
        ]
        for the_dict, the_correct_dict in zip(configurations, correct_configurations):
            self.assertDictEqual(the_dict, the_correct_dict)

        size = self.complex_expression.number_of_multiple_expressions()
        self.assertEqual(size, len(configurations))

        configurations = [
            configuration for configuration, e in self.synchronized_expression
        ]
        print(configurations)
        correct_configurations = [
            {'catalog_40': 'one', 'catalog_50': 'four'},
            {
                'catalog_60': 'two',
                'catalog_40': 'c60',
                'catalog_70': 'five',
                'catalog_50': 'c70',
            },
            {
                'catalog_60': 'three',
                'catalog_40': 'c60',
                'catalog_70': 'six',
                'catalog_50': 'c70',
            },
        ]
        for the_dict, the_correct_dict in zip(configurations, correct_configurations):
            self.assertDictEqual(the_dict, the_correct_dict)

        configurations = [
            configuration for configuration, e in self.synchronized_expression_2
        ]
        correct_configurations = [
            {'catalog_40': 'one', 'catalog_51': 'four'},
            {
                'catalog_60': 'two',
                'catalog_40': 'c60',
                'catalog_71': 'five',
                'catalog_51': 'c71',
            },
            {
                'catalog_60': 'two',
                'catalog_40': 'c60',
                'catalog_71': 'six',
                'catalog_51': 'c71',
            },
            {
                'catalog_60': 'three',
                'catalog_40': 'c60',
                'catalog_71': 'five',
                'catalog_51': 'c71',
            },
            {
                'catalog_60': 'three',
                'catalog_40': 'c60',
                'catalog_71': 'six',
                'catalog_51': 'c71',
            },
        ]
        for the_dict, the_correct_dict in zip(configurations, correct_configurations):
            self.assertDictEqual(the_dict, the_correct_dict)


    def test_selected_iterator(self):
        expression = self.catalog_1 + self.catalog_2
        selected_configurations = [
            {'catalog_1': 'g1_first', 'catalog_2': 'g2_second'},
            {'catalog_1': 'g1_first', 'catalog_2': 'g2_third'},
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_second'},
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_third'},
        ]
        iterator = ex.SelectedExpressionsIterator(
            expression,
            selected_configurations
        )
        configurations = [configuration for configuration, e in iterator]
        for the_dict, the_correct_dict in zip(configurations, selected_configurations):
            self.assertDictEqual(the_dict, the_correct_dict)

            
    def test_configure_catalogs(self):
        expression = self.catalog_1 + self.catalog_2
        a_config = {'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}
        expression.configure_catalogs(a_config)
        new_config = expression.current_configuration()
        self.assertDictEqual(a_config, new_config)

        a_config = {'catalog_4': 'one', 'catalog_7': 'five', 'catalog_5': 'c7'}
        self.complex_expression.configure_catalogs(a_config)
        new_config = self.complex_expression.current_configuration()
        self.assertDictEqual(a_config, new_config)

    def test_current_configuration(self):
        expression = self.catalog_1 + self.catalog_2
        expression.reset_expression_selection()
        config = expression.current_configuration()
        correct_config = {'catalog_1': 'g1_first', 'catalog_2': 'g2_first'}
        self.assertDictEqual(config, correct_config)
        expression.increment_selection()
        config = expression.current_configuration()
        correct_config = {'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}
        self.assertDictEqual(config, correct_config)

    def test_increment_catalog(self):
        expression = self.catalog_1 + self.catalog_2
        expression.reset_expression_selection()
        config = expression.current_configuration()
        correct_config = {'catalog_1': 'g1_first', 'catalog_2': 'g2_first'}
        self.assertDictEqual(config, correct_config)
        expression.increment_catalog('catalog_1', 1)
        config = expression.current_configuration()
        correct_config = {'catalog_1': 'g1_second', 'catalog_2': 'g2_first'}
        self.assertDictEqual(config, correct_config)
        with self.assertRaises(excep.valueOutOfRange):
            expression.increment_catalog('catalog_1', 1)

        expression.reset_expression_selection()
        expression.increment_catalog('catalog_2', 1)
        config = expression.current_configuration()
        correct_config = {'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}
        self.assertDictEqual(config, correct_config)
        expression.increment_catalog('catalog_2', -1)
        config = expression.current_configuration()
        correct_config = {'catalog_1': 'g1_first', 'catalog_2': 'g2_first'}
        self.assertDictEqual(config, correct_config)
        with self.assertRaises(excep.valueOutOfRange):
            expression.increment_catalog('catalog_2', -1)

        
if __name__ == '__main__':
    unittest.main()
