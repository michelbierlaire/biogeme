import unittest

from biogeme.catalog import Catalog, generic_alt_specific_catalogs
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta
from biogeme.segmentation import DiscreteSegmentationTuple


class TestGenericAltSpecificCatalog(unittest.TestCase):

    def setUp(self):
        # Create dummy Beta parameters
        self.beta1 = Beta('beta1', 1, None, None, 0)
        self.beta2 = Beta('beta2', 2, None, None, 0)
        self.alternatives = ('Car', 'Train')
        self.segmentations = (
            DiscreteSegmentationTuple(variable='income', mapping={0: 'low', 1: 'high'}),
        )

    def test_error_if_too_few_alternatives(self):
        with self.assertRaises(BiogemeError) as cm:
            generic_alt_specific_catalogs(
                'test',
                [self.beta1],
                ('OnlyOneAlt',),
            )
        self.assertIn('requires at least 2 alternatives', str(cm.exception))

    def test_error_if_not_list(self):
        with self.assertRaises(BiogemeError) as cm:
            generic_alt_specific_catalogs(
                'test',
                'not_a_list',
                self.alternatives,
            )
        self.assertIn('must be a list', str(cm.exception))

    def test_error_if_not_beta(self):
        with self.assertRaises(BiogemeError) as cm:
            generic_alt_specific_catalogs(
                'test',
                [self.beta1, 42],
                self.alternatives,
            )
        self.assertIn('not Beta expressions', str(cm.exception))

    def test_return_structure_without_segmentation(self):
        catalogs = generic_alt_specific_catalogs(
            'test', [self.beta1, self.beta2], self.alternatives
        )
        self.assertEqual(len(catalogs), 2)
        for item in catalogs:
            self.assertIsInstance(item, dict)
            self.assertEqual(set(item.keys()), set(self.alternatives))
            for alt, cat in item.items():
                self.assertIsInstance(cat, Catalog)
                self.assertIn('generic', [expr.name for expr in cat])
                self.assertIn('altspec', [expr.name for expr in cat])

    def test_return_structure_with_segmentation(self):
        catalogs = generic_alt_specific_catalogs(
            'test',
            [self.beta1],
            self.alternatives,
            potential_segmentations=self.segmentations,
        )
        self.assertEqual(len(catalogs), 1)
        self.assertIsInstance(catalogs[0], dict)
        for alt in self.alternatives:
            names = [expr.name for expr in catalogs[0][alt]]
            self.assertIsInstance(catalogs[0][alt], Catalog)
            self.assertIn('generic', names)
            self.assertIn('altspec', names)


if __name__ == '__main__':
    unittest.main()
