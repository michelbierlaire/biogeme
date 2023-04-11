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
import biogeme.catalog as cat
import biogeme.segmentation as seg
from biogeme.elementary_expressions import TypeOfElementaryExpression
from biogeme.configuration import Configuration, SEPARATOR, SELECTION_SEPARATOR

class TestMultipleExpressions(unittest.TestCase):
    def setUp(self):
        betas = [ex.Beta(f'beta{i}', i * 0.1, None, None, 0) for i in range(10)]
        g1_first = cat.NamedExpression(name='g1_first', expression=betas[0])
        g1_second = cat.NamedExpression(name='g1_second', expression=betas[1])
        self.g1_tuple = (g1_first, g1_second)
        self.catalog_1 = cat.Catalog('catalog_1', self.g1_tuple)
        self.assertEqual(self.catalog_1.current_index, 0)

        g2_zero = cat.NamedExpression(name='one', expression=ex.Numeric(1))
        g2_first = cat.NamedExpression(name='g2_first', expression=betas[2])
        g2_second = cat.NamedExpression(name='g2_second', expression=betas[3])
        g2_third = cat.NamedExpression(name='g2_third', expression=betas[4])
        g2_fourth = cat.NamedExpression(name='g2_fourth', expression=self.catalog_1)
        g2_tuple = (g2_first, g2_second, g2_third)
        self.catalog_2 = cat.Catalog('catalog_2', g2_tuple)
        self.assertEqual(self.catalog_2.current_index, 0)

        g3_tuple = (g2_zero, g2_first, g2_second, g2_third, g2_fourth)
        self.catalog_3 = cat.Catalog('catalog_3', g3_tuple)

        one = cat.NamedExpression(name='one', expression=ex.Numeric(1))
        two = cat.NamedExpression(name='two', expression=ex.Numeric(2))
        three = cat.NamedExpression(name='three', expression=ex.Numeric(3))
        four = cat.NamedExpression(name='four', expression=ex.Numeric(4))
        five = cat.NamedExpression(name='five', expression=ex.Numeric(5))
        six = cat.NamedExpression(name='six', expression=ex.Numeric(6))

        c6_tuple = (two, three)
        catalog_6 = cat.Catalog('catalog_6', c6_tuple)
        c6 = cat.NamedExpression(name='c6', expression=catalog_6)
        c6_tuple = (five, six)
        catalog_7 = cat.Catalog('catalog_7', c6_tuple)
        c7 = cat.NamedExpression(name='c7', expression=catalog_7)
        c4_tuple = (one, c6)
        catalog_4 = cat.Catalog('catalog_4', c4_tuple)
        c5_tuple = (four, c7)
        catalog_5 = cat.Catalog('catalog_5', c5_tuple)
        self.complex_expression = catalog_4 + catalog_5

        # Same expression as above, where catalogs are synchronized
        c60_tuple = (two, three)
        catalog_60 = cat.Catalog('catalog_60', c60_tuple)
        c60 = cat.NamedExpression(name='c60', expression=catalog_60)
        c70_tuple = (five, six)
        catalog_70 = cat.SynchronizedCatalog('catalog_70', c70_tuple, catalog_60)
        catalog_71 = cat.Catalog('catalog_71', c70_tuple)
        c70 = cat.NamedExpression(name='c70', expression=catalog_70)
        c71 = cat.NamedExpression(name='c71', expression=catalog_71)
        c40_tuple = (one, c60)
        catalog_40 = cat.Catalog('catalog_40', c40_tuple)
        c50_tuple = (four, c70)
        c51_tuple = (four, c71)
        catalog_50 = cat.SynchronizedCatalog('catalog_50', c50_tuple, catalog_40)
        catalog_51 = cat.SynchronizedCatalog('catalog_51', c51_tuple, catalog_40)
        self.synchronized_expression = catalog_40 + catalog_50
        self.synchronized_expression_2 = catalog_40 + catalog_51

        self.wrong_expression = catalog_6 + catalog_6

        c81 = cat.NamedExpression(name='c81', expression=self.catalog_1)
        c82 = cat.NamedExpression(name='c82', expression=self.catalog_1)
        self.catalog_8 = cat.Catalog('catalog_of_catalogs', (c81, c82) )
        
    def test_ctor(self):
        empty_tuple = tuple()
        with self.assertRaises(excep.biogemeError):
            _ = cat.Catalog('the_name', empty_tuple)

        incorrect = cat.NamedExpression(
            name='incorrect', expression='not_an_expression'
        )
        wrong_tuple = (incorrect,)
        with self.assertRaises(excep.biogemeError):
            _ = cat.Catalog('the_name', wrong_tuple)

        with self.assertRaises(excep.biogemeError):
            _ = cat.Catalog(f'the_name{SEPARATOR}', self.g1_tuple)

        with self.assertRaises(excep.biogemeError):
            _ = cat.Catalog(f'the_name{SELECTION_SEPARATOR}', self.g1_tuple)
            

        
    def test_names(self):
        result = [e.name for e in self.catalog_1.list_of_named_expressions]
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

        size = self.catalog_8.catalog_size()
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

        the_dict = self.catalog_8.dict_of_catalogs()
        correct_keys = ['catalog_1', 'catalog_of_catalogs']
        self.assertListEqual(list(the_dict.keys()), correct_keys)
        
    def test_number_of_multiple_expressions(self):
        expression = self.catalog_1 + self.catalog_2
        the_number = expression.number_of_multiple_expressions()
        correct_number = 6
        self.assertEqual(the_number, correct_number)

        expression.maximum_number_of_configurations = 2
        the_number = expression.number_of_multiple_expressions()
        self.assertIsNone(the_number)
        
        another_expression = ex.Numeric(2) + self.catalog_3
        the_number = another_expression.number_of_multiple_expressions()
        correct_number = 6
        self.assertEqual(the_number, correct_number)

        the_number = self.catalog_8.number_of_multiple_expressions()
        correct_number = 4
        self.assertEqual(the_number, correct_number)
        

    def test_synchronized(self):
        expression = self.synchronized_expression
        conf_1 = Configuration.from_dict({'catalog_40': 'one'})
        conf_2 = Configuration.from_dict({'catalog_60': 'two', 'catalog_40': 'c60'})
        conf_3 = Configuration.from_dict({'catalog_60': 'three', 'catalog_40': 'c60'})

        check_1 = Configuration.from_dict(
            {
                'catalog_40': 'one',
                'catalog_50': 'four',
            }
        )

        check_2 = Configuration.from_dict(
            {
                'catalog_40': 'c60',
                'catalog_50': 'c70',
                'catalog_60': 'two',
                'catalog_70': 'five'
            }
        )
        check_3 = Configuration.from_dict(
            {
                'catalog_40': 'c60',
                'catalog_50': 'c70',
                'catalog_60': 'three',
                'catalog_70': 'six'
            }
        )
        expression.configure_catalogs(conf_1)
        actual_1 = expression.current_configuration(includes_controlled_catalogs=True)
        self.assertEqual(check_1, actual_1)

        expression.configure_catalogs(conf_2)
        actual_2 = expression.current_configuration(includes_controlled_catalogs=True)
        self.assertEqual(check_2, actual_2)

        expression.configure_catalogs(conf_3)
        actual_3 = expression.current_configuration(includes_controlled_catalogs=True)
        self.assertEqual(check_3, actual_3)
        
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
        expression.maximum_number_of_configurations = 2
        with self.assertRaises(excep.valueOutOfRange):
            for e in expression:
                pass
        expression.maximum_number_of_configurations = 100
        configurations = {e.current_configuration() for e in expression}
        correct_configurations = {
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_first'}),
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}),
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_third'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_first'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_second'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_third'}),
        }
        self.assertSetEqual(configurations, correct_configurations)

        size = expression.number_of_multiple_expressions()
        self.assertEqual(size, len(configurations))

        another_expression = ex.Numeric(2) + self.catalog_3
        configurations = {
            e.current_configuration() for e in another_expression
        }
        correct_configurations = {
            Configuration.from_dict({'catalog_3': 'one'}),
            Configuration.from_dict({'catalog_3': 'g2_first'}),
            Configuration.from_dict({'catalog_3': 'g2_second'}),
            Configuration.from_dict({'catalog_3': 'g2_third'}),
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_3': 'g2_fourth'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_3': 'g2_fourth'}),
        }
        self.assertSetEqual(configurations, correct_configurations)

        size = another_expression.number_of_multiple_expressions()
        self.assertEqual(size, len(configurations))

        configurations = {e.current_configuration() for e in self.complex_expression}
        correct_configurations = {
            Configuration.from_dict(
                {'catalog_4': 'one', 'catalog_5': 'four'}
            ),
            Configuration.from_dict(
                {'catalog_4': 'one', 'catalog_7': 'five', 'catalog_5': 'c7'}
            ),
            Configuration.from_dict(
                {'catalog_4': 'one', 'catalog_7': 'six', 'catalog_5': 'c7'}
            ),
            Configuration.from_dict(
                {'catalog_6': 'two', 'catalog_4': 'c6', 'catalog_5': 'four'}
            ),
            Configuration.from_dict(
                {
                    'catalog_6': 'two',
                    'catalog_4': 'c6',
                    'catalog_7': 'five',
                    'catalog_5': 'c7',
                }
            ),
            Configuration.from_dict(
                {
                    'catalog_6': 'two',
                    'catalog_4': 'c6',
                    'catalog_7': 'six',
                    'catalog_5': 'c7',
                }
            ),
            Configuration.from_dict(
                {'catalog_6': 'three', 'catalog_4': 'c6', 'catalog_5': 'four'}
            ),
            Configuration.from_dict(
                {
                    'catalog_6': 'three',
                    'catalog_4': 'c6',
                    'catalog_7': 'five',
                    'catalog_5': 'c7',
                }
            ),
            Configuration.from_dict(
                {
                    'catalog_6': 'three',
                    'catalog_4': 'c6',
                    'catalog_7': 'six',
                    'catalog_5': 'c7',
                }
            ),
        }
        self.assertSetEqual(configurations, correct_configurations)

        size = self.complex_expression.number_of_multiple_expressions()
        self.assertEqual(size, len(configurations))

        configurations = {
            e.current_configuration() for e in self.synchronized_expression
        }

        correct_configurations = {
            Configuration.from_dict({'catalog_40': 'one'}),
            Configuration.from_dict({'catalog_60': 'two', 'catalog_40': 'c60'}),
            Configuration.from_dict({'catalog_60': 'three', 'catalog_40': 'c60'}),
        }
        self.assertSetEqual(configurations, correct_configurations)

        configurations = {
            e.current_configuration() for e in self.synchronized_expression_2
        }
        correct_configurations = {
            Configuration.from_dict({'catalog_40': 'one'}),
            Configuration.from_dict({'catalog_60': 'two', 'catalog_40': 'c60', 'catalog_71': 'five'}),
            Configuration.from_dict({'catalog_60': 'two', 'catalog_40': 'c60', 'catalog_71': 'six'}),
            Configuration.from_dict({'catalog_60': 'three', 'catalog_40': 'c60', 'catalog_71': 'five'}),
            Configuration.from_dict({'catalog_60': 'three', 'catalog_40': 'c60', 'catalog_71': 'six'}),
        }
        self.assertSetEqual(configurations, correct_configurations)

        configurations = {
            e.current_configuration() for e in self.catalog_8
        }
        correct_configurations = {
            Configuration.from_string('catalog_1:g1_first;catalog_of_catalogs:c81'),
            Configuration.from_string('catalog_1:g1_second;catalog_of_catalogs:c81'),
            Configuration.from_string('catalog_1:g1_first;catalog_of_catalogs:c82'),
            Configuration.from_string('catalog_1:g1_second;catalog_of_catalogs:c82')
        }
        self.assertSetEqual(configurations, correct_configurations)

    def test_selected_iterator(self):
        expression = self.catalog_1 + self.catalog_2
        selected_configurations = [
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}),
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_third'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_second'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_third'}),
        ]
        iterator = ex.SelectedExpressionsIterator(expression, selected_configurations)
        configurations = [e.current_configuration() for e in iterator]
        self.assertListEqual(selected_configurations, configurations)

    def test_configure_catalogs(self):
        expression = self.catalog_1 + self.catalog_2
        a_config = Configuration.from_dict(
            {'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}
        )
        the_config_id = a_config.get_string_id()
        test_config = Configuration.from_string(the_config_id)
        self.assertEqual(a_config, test_config)
        self.assertEqual(the_config_id, test_config.get_string_id())
        expression.configure_catalogs(a_config)
        new_config = expression.current_configuration()
        self.assertEqual(a_config, new_config)

        a_config = Configuration.from_dict(
            {'catalog_4': 'one', 'catalog_7': 'five', 'catalog_5': 'c7'}
        )
        self.complex_expression.configure_catalogs(a_config)
        new_config = self.complex_expression.current_configuration()
        self.assertEqual(a_config, new_config)

    def test_segmentation(self):
        beta = ex.Beta('beta', 0, None, None, 0)
        var1 = ex.Variable('var1')
        mapping1 = {0: 'zero', 1: 'one'}
        segmentation1 = seg.DiscreteSegmentationTuple(variable=var1, mapping=mapping1)
        var2 = ex.Variable('var2')
        mapping2 = {0: 'no', 1: 'yes'}
        segmentation2 = seg.DiscreteSegmentationTuple(variable=var2, mapping=mapping2)
        tuple_of_segmentations = (
            segmentation1,
            segmentation2,
        )
        the_catalog = cat.segmentation_catalog(
            beta,
            tuple_of_segmentations,
            maximum_number=2,
        )
        configurations = {c.current_configuration() for c in the_catalog}
        correct_configuration = {
            Configuration.from_dict({'segmented_beta': 'beta (no seg.)'}),
            Configuration.from_dict({'segmented_beta': 'var2'}),
            Configuration.from_dict({'segmented_beta': 'var1'}),
            Configuration.from_dict({'segmented_beta': 'var1-var2'}),
        }
        self.assertSetEqual(configurations, correct_configuration)

        for c in the_catalog:
            the_betas = c.dict_of_elementary_expression(TypeOfElementaryExpression.FREE_BETA)
        
        the_catalog = cat.segmentation_catalog(
            beta,
            tuple_of_segmentations,
            maximum_number=1,
        )
        configurations = {c.current_configuration() for c in the_catalog}
        correct_configuration = {
            Configuration.from_dict({'segmented_beta': 'beta (no seg.)'}),
            Configuration.from_dict({'segmented_beta': 'var2'}),
            Configuration.from_dict({'segmented_beta': 'var1'}),
        }
        self.assertSetEqual(configurations, correct_configuration)
            
    def test_set_of_configurations(self):
        expression = self.catalog_1 + self.catalog_2
        expression.maximum_number_of_configurations = 2
        the_set = expression.set_of_configurations()
        self.assertIsNone(the_set)
        expression.maximum_number_of_configurations = 100
        the_set = expression.set_of_configurations()
        the_correct_set = {
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_first'}),
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}),
            Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_third'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_first'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_second'}),
            Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_third'}),
        }
        self.assertSetEqual(the_set, the_correct_set)
        the_size = expression.number_of_multiple_expressions()
        self.assertEqual(the_size, len(the_set))
        
        the_set = self.synchronized_expression.set_of_configurations()
        the_correct_set = {
            Configuration.from_dict({'catalog_40': 'one'}),
            Configuration.from_dict({'catalog_60': 'two', 'catalog_40': 'c60'}),
            Configuration.from_dict({'catalog_60': 'three', 'catalog_40': 'c60'}),
        }
        self.assertSetEqual(the_set, the_correct_set)
        the_size = self.synchronized_expression.number_of_multiple_expressions()
        self.assertEqual(the_size, len(the_set))

        the_set = self.catalog_1.set_of_configurations()
        the_correct_set = {
            Configuration.from_dict({'catalog_1': 'g1_first'}),
            Configuration.from_dict({'catalog_1': 'g1_second'}),
        }
        self.assertSetEqual(the_set, the_correct_set)
        the_size = self.catalog_1.number_of_multiple_expressions()
        self.assertEqual(the_size, len(the_set))

    def test_modify_catalogs_a(self):
        expression = self.catalog_3
        #correct_configurations = {
        #    Configuration.from_dict({'catalog_3': 'one'}),
        #    Configuration.from_dict({'catalog_3': 'g2_first'}),
        #    Configuration.from_dict({'catalog_3': 'g2_second'}),
        #    Configuration.from_dict({'catalog_3': 'g2_third'}),
        #    Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_3': 'g2_fourth'}),
        #    Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_3': 'g2_fourth'}),
        #}

        a_config = Configuration.from_dict({'catalog_3': 'g2_first'})
        expression.configure_catalogs(a_config)
        check_config = expression.current_configuration()
        self.assertEqual(a_config, check_config)

        expression.modify_catalogs({'catalog_3'}, -1, True)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_3': 'one'}
        )
        self.assertEqual(correct_config, check_config)

        expression.modify_catalogs({'catalog_3', 'catalog_1'}, 3, True)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_3': 'g2_third'}
        )
        self.assertEqual(correct_config, check_config)
        
        expression.modify_catalogs({'catalog_3'}, 1, True)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_3': 'g2_fourth', 'catalog_1': 'g1_second'}
        )
        self.assertEqual(correct_config, check_config)
        
    def test_modify_catalogs_b(self):
        expression = self.catalog_1 + self.catalog_2
        
        # Set of configurations for this expressions
        #the_correct_set = {
        #    Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_first'}),
        #    Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_second'}),
        #    Configuration.from_dict({'catalog_1': 'g1_first', 'catalog_2': 'g2_third'}),
        #    Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_first'}),
        #    Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_second'}),
        #    Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_third'}),
        #}
        a_config = Configuration.from_dict({'catalog_1': 'g1_second', 'catalog_2': 'g2_first'})
        expression.configure_catalogs(a_config)
        check_config = expression.current_configuration()
        self.assertEqual(a_config, check_config)
        
        expression.modify_catalogs({'catalog_2'}, 1, True)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_second'}
        )
        self.assertEqual(correct_config, check_config)

        expression.modify_catalogs({'catalog_2'}, 3, False)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_third'}
        )
        self.assertEqual(correct_config, check_config)

        expression.modify_catalogs({'catalog_2'}, -3, True)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_third'}
        )
        self.assertEqual(correct_config, check_config)

        expression.modify_catalogs({'catalog_1'}, -1, True)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_1': 'g1_first', 'catalog_2': 'g2_third'}
        )
        self.assertEqual(correct_config, check_config)

        expression.modify_catalogs({'catalog_1', 'catalog_2'}, -1, True)
        check_config = expression.current_configuration()
        correct_config = Configuration.from_dict(
            {'catalog_1': 'g1_second', 'catalog_2': 'g2_second'}
        )
        self.assertEqual(correct_config, check_config)
        
    def test_modify_catalogs_c(self):
        expression = self.synchronized_expression
        conf_1 = Configuration.from_dict({'catalog_40': 'one'})
        conf_2 = Configuration.from_dict({'catalog_60': 'two', 'catalog_40': 'c60'})
        conf_3 = Configuration.from_dict({'catalog_60': 'three', 'catalog_40': 'c60'})

        check_1 = Configuration.from_dict(
            {
                'catalog_40': 'one',
                'catalog_50': 'four',
            }
        )

        check_2 = Configuration.from_dict(
            {
                'catalog_40': 'c60',
                'catalog_50': 'c70',
                'catalog_60': 'two',
                'catalog_70': 'five'
            }
        )
        check_3 = Configuration.from_dict(
            {
                'catalog_40': 'c60',
                'catalog_50': 'c70',
                'catalog_60': 'three',
                'catalog_70': 'six'
            }
        )
        
        expression.configure_catalogs(conf_1)
        # This should not do anything, as catalog_50 is controlled
        expression.modify_catalogs({'catalog_50'}, -1, True)
        check_config = expression.current_configuration(includes_controlled_catalogs=True)
        self.assertEqual(check_config, check_1)
        expression.modify_catalogs({'catalog_40'}, -1, True)
        check_config = expression.current_configuration(includes_controlled_catalogs=True)
        self.assertEqual(check_config, check_2)
        expression.modify_catalogs({'catalog_40', 'catalog_60'}, 1, True)
        check_config = expression.current_configuration(includes_controlled_catalogs=True)
        self.assertEqual(check_config, check_1)
        expression.modify_catalogs({'catalog_40'}, 1, False)
        check_config = expression.current_configuration(includes_controlled_catalogs=True)
        self.assertEqual(check_config, check_3)
        
if __name__ == '__main__':
    unittest.main()
