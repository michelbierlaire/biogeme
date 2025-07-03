import unittest

from biogeme.catalog import Catalog, segmentation_catalogs
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, SELECTION_SEPARATOR, SEPARATOR
from biogeme.segmentation import DiscreteSegmentationTuple


class TestSegmentationCatalogs(unittest.TestCase):

    def setUp(self):
        self.beta = Beta('test_param', 1.0, None, None, 0)
        self.beta_list = [self.beta]
        self.valid_segmentation = DiscreteSegmentationTuple(
            variable='group', mapping={1: 'A', 2: 'B'}
        )

    def test_valid_catalogs_generation(self):
        catalogs = segmentation_catalogs(
            generic_name='test',
            beta_parameters=self.beta_list,
            potential_segmentations=(self.valid_segmentation,),
            maximum_number=1,
        )
        self.assertIsInstance(catalogs, list)
        self.assertTrue(all(isinstance(cat, Catalog) for cat in catalogs))
        self.assertEqual(len(catalogs), 1)
        self.assertTrue(catalogs[0].name.startswith('segmented_'))

    def test_invalid_beta_parameters_type(self):
        with self.assertRaisesRegex(
            BiogemeError, 'A list is expected for beta_parameters'
        ):
            segmentation_catalogs(
                generic_name='test',
                beta_parameters='not_a_list',
                potential_segmentations=(self.valid_segmentation,),
                maximum_number=1,
            )

    def test_segment_name_with_separator_raises(self):
        bad_segmentation = DiscreteSegmentationTuple(
            variable='group', mapping={1: f'bad{SEPARATOR}name'}
        )
        with self.assertRaisesRegex(BiogemeError, 'Invalid segment name for variable'):
            segmentation_catalogs(
                generic_name='test',
                beta_parameters=self.beta_list,
                potential_segmentations=(bad_segmentation,),
                maximum_number=1,
            )

    def test_segment_name_with_selection_separator_raises(self):
        bad_segmentation = DiscreteSegmentationTuple(
            variable='group', mapping={1: f'bad{SELECTION_SEPARATOR}name'}
        )
        with self.assertRaisesRegex(BiogemeError, 'Invalid segment name for variable'):
            segmentation_catalogs(
                generic_name='test',
                beta_parameters=self.beta_list,
                potential_segmentations=(bad_segmentation,),
                maximum_number=1,
            )

    def test_no_segmentation_case_included(self):
        catalogs = segmentation_catalogs(
            generic_name='test',
            beta_parameters=self.beta_list,
            potential_segmentations=(self.valid_segmentation,),
            maximum_number=0,
        )
        names = [named.name for named in catalogs[0].named_expressions]
        self.assertIn('no_seg', names)


if __name__ == '__main__':
    unittest.main()
