import unittest
from biogeme.expressions import Beta, Variable
from biogeme.exceptions import BiogemeError
from biogeme.segmentation import DiscreteSegmentationTuple
from biogeme.catalog import (
    Catalog,
    SegmentedParameters,
    segmentation_catalogs,
    generic_alt_specific_catalogs,
)
from biogeme.expressions import NamedExpression
from biogeme.configuration import (
    SelectionTuple,
    Configuration,
)


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.expression1 = Beta('expr1', 0, None, None, 0)
        self.expression2 = Beta('expr2', 0, None, None, 0)
        self.expression3 = Beta('expr3', 0, None, None, 0)

        self.named_expression1 = NamedExpression('expression1', self.expression1)
        self.named_expression2 = NamedExpression('expression2', self.expression2)
        self.named_expression3 = NamedExpression('expression3', self.expression3)

        self.catalog = Catalog(
            'my_catalog',
            [self.named_expression1, self.named_expression2, self.named_expression3],
        )

        self.another_catalog = Catalog(
            'another_catalog',
            [self.named_expression1, self.named_expression2, self.named_expression3],
        )

    def test_catalog_creation(self):
        # Test catalog creation with valid input
        catalog = Catalog(
            'test_catalog', [self.named_expression1, self.named_expression2]
        )

        self.assertEqual(catalog.name, 'test_catalog')
        self.assertEqual(len(catalog.named_expressions), 2)
        self.assertEqual(catalog.catalog_size(), 2)
        self.assertEqual(catalog.named_expressions[0].name, 'expression1')
        self.assertEqual(catalog.named_expressions[1].name, 'expression2')
        self.assertIs(catalog.named_expressions[0].expression, self.expression1)
        self.assertIs(catalog.named_expressions[1].expression, self.expression2)

        # Test catalog creation with an empty list
        with self.assertRaises(BiogemeError):
            Catalog('empty_catalog', [])

        # Trying to control with a relevant Catalog
        other_catalog = Catalog(
            catalog_name='other_catalog',
            named_expressions=[self.named_expression1, self.named_expression2],
            controlled_by=catalog.controlled_by,
        )
        self.assertIs(other_catalog.controlled_by, catalog.controlled_by)

        # Trying to control with an irrelevant Catalog
        with self.assertRaises(BiogemeError):
            other_catalog = Catalog(
                catalog_name='other_catalog',
                named_expressions=[self.named_expression1, self.named_expression2],
                controlled_by=self.catalog.controlled_by,
            )

    def test_catalog_from_dict(self):
        # Test catalog creation from a dictionary
        dict_of_expressions = {'expr1': self.expression1, 'expr2': self.expression2}
        catalog = Catalog.from_dict('test_catalog', dict_of_expressions)

        self.assertEqual(catalog.name, 'test_catalog')
        self.assertEqual(len(catalog.named_expressions), 2)
        self.assertEqual(catalog.named_expressions[0].name, 'expr1')
        self.assertEqual(catalog.named_expressions[1].name, 'expr2')
        self.assertIs(catalog.named_expressions[0].expression, self.expression1)
        self.assertIs(catalog.named_expressions[1].expression, self.expression2)
        self.assertEqual(catalog.catalog_size(), 2)

    def test_set_of_configurations(self):
        config_1 = Configuration(
            [SelectionTuple(controller='my_catalog', selection='expression1')]
        )
        config_2 = Configuration(
            [SelectionTuple(controller='my_catalog', selection='expression2')]
        )
        config_3 = Configuration(
            [SelectionTuple(controller='my_catalog', selection='expression3')]
        )
        expected_set = {
            config_1,
            config_2,
            config_3,
        }
        the_set = self.catalog.set_of_configurations()
        self.assertSetEqual(the_set, expected_set)

    def test_selected_expression(self):
        selected_expression = self.catalog.selected_expression()
        self.assertEqual(selected_expression, self.expression1)

    def test_selected_name(self):
        selected_name = self.catalog.selected_name()
        self.assertEqual(selected_name, 'expression1')

    def test_current_configuration(self):
        configuration = self.catalog.current_configuration()
        self.assertEqual(len(configuration.selections), 1)
        self.assertEqual(configuration.selections[0].controller, 'my_catalog')
        self.assertEqual(configuration.selections[0].selection, 'expression1')

    def test_iterator(self):
        my_expression = self.catalog + self.another_catalog
        configurations = {e.current_configuration() for e in my_expression}
        expected_configurations = {
            Configuration.from_string(
                'another_catalog:expression1;my_catalog:expression1'
            ),
            Configuration.from_string(
                'another_catalog:expression1;my_catalog:expression2'
            ),
            Configuration.from_string(
                'another_catalog:expression1;my_catalog:expression3'
            ),
            Configuration.from_string(
                'another_catalog:expression2;my_catalog:expression1'
            ),
            Configuration.from_string(
                'another_catalog:expression2;my_catalog:expression2'
            ),
            Configuration.from_string(
                'another_catalog:expression2;my_catalog:expression3'
            ),
            Configuration.from_string(
                'another_catalog:expression3;my_catalog:expression1'
            ),
            Configuration.from_string(
                'another_catalog:expression3;my_catalog:expression2'
            ),
            Configuration.from_string(
                'another_catalog:expression3;my_catalog:expression3'
            ),
        }
        self.assertSetEqual(configurations, expected_configurations)


class SegmentationCatalogTest(unittest.TestCase):
    def setUp(self):
        self.beta_parameter_1 = Beta('beta_1', 0, None, 10, 0)
        self.beta_parameter_2 = Beta('beta_2', 0.5, -10, 10, 0)
        self.variable_x = Variable('x')
        self.variable_y = Variable('y')
        self.variable_z = Variable('z')
        self.segmentation1 = DiscreteSegmentationTuple(
            self.variable_x, {0: 'zero', 1: 'one'}
        )
        self.segmentation2 = DiscreteSegmentationTuple(
            self.variable_y, {0: 'zero', 1: 'one'}
        )
        self.segmentation3 = DiscreteSegmentationTuple(
            self.variable_z, {0: 'zero', 1: 'one'}
        )
        self.potential_segmentations = (
            self.segmentation1,
            self.segmentation2,
            self.segmentation3,
        )

    def test_wrong_segment_name(self):
        wrong_segmentation_1 = DiscreteSegmentationTuple(
            self.variable_z, {0: 'zero:', 1: 'one'}
        )
        potential_segmentations_1 = (wrong_segmentation_1,)
        with self.assertRaises(BiogemeError):
            _ = segmentation_catalogs(
                generic_name='wrong_segmentation',
                beta_parameters=[self.beta_parameter_1],
                potential_segmentations=potential_segmentations_1,
                maximum_number=1,
            )

        wrong_segmentation_2 = DiscreteSegmentationTuple(
            self.variable_z, {0: 'zero', 1: 'on;e'}
        )
        potential_segmentations_2 = (wrong_segmentation_2,)
        with self.assertRaises(BiogemeError):
            _ = segmentation_catalogs(
                generic_name='wrong_segmentation',
                beta_parameters=[self.beta_parameter_1],
                potential_segmentations=potential_segmentations_2,
                maximum_number=1,
            )

    def test_segmentation_catalog_no_sync(self):
        catalogs = segmentation_catalogs(
            generic_name='segmentated_beta',
            beta_parameters=[self.beta_parameter_1],
            potential_segmentations=self.potential_segmentations,
            maximum_number=2,
        )
        self.assertIsInstance(catalogs[0], Catalog)
        self.assertEqual(catalogs[0].name, 'segmented_beta_1')
        self.assertEqual(len(catalogs[0].named_expressions), 7)

    def test_segmentation_catalog_with_sync(self):
        catalogs = segmentation_catalogs(
            generic_name='segmented_beta',
            beta_parameters=[self.beta_parameter_1, self.beta_parameter_2],
            potential_segmentations=self.potential_segmentations,
            maximum_number=2,
        )
        self.assertIsInstance(catalogs[0], Catalog)
        self.assertIsInstance(catalogs[1], Catalog)
        self.assertEqual(catalogs[0].name, 'segmented_beta_1')
        self.assertEqual(catalogs[1].name, 'segmented_beta_2')
        self.assertEqual(len(catalogs[0].named_expressions), 7)
        self.assertIs(catalogs[0].controlled_by, catalogs[1].controlled_by)

    def test_segmentation_catalog_no_segmentations(self):
        catalogs = segmentation_catalogs(
            generic_name='segmented_beta',
            beta_parameters=[self.beta_parameter_1],
            potential_segmentations=(),
            maximum_number=2,
        )
        self.assertIsInstance(catalogs[0], Catalog)
        self.assertEqual(catalogs[0].name, 'segmented_beta_1')
        self.assertEqual(len(catalogs[0].named_expressions), 1)


class TestSegmentedParameters(unittest.TestCase):
    def setUp(self):
        self.beta_parameters = [
            Beta('beta_time', 0, None, None, 0),
            Beta('beta_cost', 0, None, None, 0),
        ]
        self.alternatives = ['train', 'car', 'sm']
        self.segmented_parameters = SegmentedParameters(
            beta_parameters=self.beta_parameters,
            alternatives=self.alternatives,
        )

    def test_constructor(self):
        self.assertEqual(len(self.segmented_parameters.all_parameters), 2 * 4)
        self.assertEqual(len(self.segmented_parameters.beta_parameters), 2)
        self.assertEqual(len(self.segmented_parameters.alternatives), 3)


class TestGenericAltSpecificCatalog(unittest.TestCase):
    def test_generic_alt_specific_catalog(self):
        coefficient = Beta(
            name='coefficient', value=1.0, lowerbound=None, upperbound=None, status=0
        )
        alternatives = ('A', 'B', 'C')

        result = generic_alt_specific_catalogs(
            generic_name='coefficient',
            beta_parameters=[coefficient],
            alternatives=alternatives,
        )

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], dict)
        self.assertEqual(len(result[0]), len(alternatives))

        for alternative, catalog in result[0].items():
            self.assertIsInstance(catalog, Catalog)
            for named_expression in catalog.named_expressions:
                if named_expression.name == 'generic':
                    self.assertEqual(named_expression.expression, coefficient)
                elif named_expression.name == 'altspec':
                    beta = named_expression.expression
                    self.assertIsInstance(beta, Beta)
                    self.assertEqual(beta.name, f'{coefficient.name}_{alternative}')
                    self.assertEqual(beta.initValue, coefficient.initValue)
                    self.assertEqual(beta.lb, coefficient.lb)
                    self.assertEqual(beta.ub, coefficient.ub)
                    self.assertEqual(beta.status, coefficient.status)
                else:
                    self.fail(f'Unknown expression {named_expression.name}')

    def test_generic_alt_specific_catalog_invalid_coefficient(self):
        coefficient = 1.0  # Invalid coefficient, not a Beta object
        alternatives = ('A', 'B', 'C')

        with self.assertRaises(BiogemeError):
            generic_alt_specific_catalogs(
                generic_name='any',
                beta_parameters=[coefficient],
                alternatives=alternatives,
            )

    def test_generic_alt_specific_catalog_insufficient_alternatives(self):
        coefficient = Beta(
            name='coefficient', value=1.0, lowerbound=None, upperbound=None, status=0
        )
        alternatives = ('A',)

        with self.assertRaises(BiogemeError):
            generic_alt_specific_catalogs(
                generic_name='any',
                beta_parameters=[coefficient],
                alternatives=alternatives,
            )


if __name__ == '__main__':
    unittest.main()
