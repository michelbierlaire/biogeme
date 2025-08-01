"""
Test the database module

:author: Michel Bierlaire
:data: Wed Apr 29 18:02:59 2020

"""

import unittest

import numpy as np
import pandas as pd

from biogeme.database import Database, sample_with_replacement
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Variable
from biogeme.segmentation import DiscreteSegmentationTuple
from test_data import (
    getData,
)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        np.random.seed(90267)
        self.myData1 = getData(1)
        self.myPanelData = getData(3)
        self.Variable1 = Variable('Variable1')
        self.Variable2 = Variable('Variable2')
        self.Av1 = Variable('Av1')
        self.Av2 = Variable('Av2')
        self.Av3 = Variable('Av3')
        self.Choice = Variable('Choice')

    def test_empty_database(self):
        df = pd.DataFrame()
        with self.assertRaises(BiogemeError):
            _ = Database('name', df)

    def test_bad_values(self):
        with self.assertRaises(BiogemeError):
            df = getData(0)

    def test_check_segmentation(self):
        the_data = getData(1)
        correct_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3'}
        correct_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=correct_mapping
        )
        # Successful check
        output = the_data.verify_segmentation(correct_segmentation)
        incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2'}
        incorrect_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=incorrect_mapping
        )
        with self.assertRaises(BiogemeError):
            the_data.verify_segmentation(incorrect_segmentation)

        another_incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 4: 'Does not exist'}
        another_incorrect_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=another_incorrect_mapping
        )
        with self.assertRaises(BiogemeError):
            the_data.verify_segmentation(another_incorrect_segmentation)

    def test_scale_column(self):
        self.myData1.scale_column('Variable1', 100)
        the_result = list(self.myData1.dataframe['Variable1'])
        expected_result = [100, 200, 300, 400, 500]
        self.assertListEqual(the_result, expected_result)

    def test_suggest_scaling(self):
        result = self.myData1.suggest_scaling()
        result.reset_index(inplace=True, drop=True)
        expected_result = pd.DataFrame(
            {
                'Column': ['Variable2'],
                'Scale': [0.01],
                'Largest': [50.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result, check_index_type=False)
        result = self.myData1.suggest_scaling(columns=['Exclude', 'Variable2'])
        result.reset_index(inplace=True, drop=True)
        pd.testing.assert_frame_equal(result, expected_result, check_index_type=False)
        with self.assertRaises(BiogemeError):
            _ = self.myData1.suggest_scaling(columns=['wrong_name'])

    def test_addColumn(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        expression = Variable2 * Variable1
        the_values = [10, 40, 90, 160, 250]
        self.myData1.add_column(column='NewVariable', values=the_values)
        the_list = self.myData1.dataframe['NewVariable'].tolist()
        self.assertListEqual(the_list, the_values)
        with self.assertRaises(ValueError):
            result = self.myData1.add_column(column='Variable1', values=the_values)
        database = getData(1)
        database._df = database.dataframe[0:0]
        with self.assertRaises(ValueError):
            result = database.add_column(column='NewVariable', values=the_values)

    def test_DefineVariable(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        expression = Variable2 * Variable1
        result = self.myData1.define_variable('NewVariable', expression)
        theList = self.myData1.dataframe['NewVariable'].tolist()
        self.assertListEqual(theList, [10, 40, 90, 160, 250])
        with self.assertRaises(ValueError):
            result = self.myData1.define_variable('Variable1', expression)
        database = getData(1)
        database._df = database.dataframe[0:0]
        with self.assertRaises(BiogemeError):
            result = database.define_variable('NewVariable', expression)

    def test_remove(self):
        Exclude = Variable('Exclude')
        d = getData(1)
        d.remove(Exclude)
        rows, _ = d.dataframe.shape
        self.assertEqual(rows, 3)
        # This should not remove anything
        d.remove(0)
        rows, _ = d.dataframe.shape
        self.assertEqual(rows, 3)

    def test_sampleWithReplacement(self):
        res1 = sample_with_replacement(df=self.myData1.dataframe)
        res2 = sample_with_replacement(df=self.myData1.dataframe, size=12)
        dim1 = res1.shape
        dim2 = res2.shape
        self.assertTupleEqual(dim1, (5, 8))
        self.assertTupleEqual(dim2, (12, 8))

    def test_panel(self):
        # Data is not considered panel yet
        should_be_false = self.myPanelData.is_panel()
        self.myPanelData.panel('Person')
        should_be_true = self.myPanelData.is_panel()
        self.assertTrue(should_be_true)
        self.assertFalse(should_be_false)

    def test_print(self):
        result = str(self.myData1)[0:23]
        expected_result = 'biogeme database test_1'
        self.assertEqual(result, expected_result)
        self.myPanelData.panel('Person')
        result = str(self.myPanelData)[0:23]
        expected_result = 'biogeme database test_3'
        self.assertEqual(result, expected_result)

    def test_getNumberOfObservations(self):
        self.myPanelData.panel('Person')
        self.assertEqual(self.myData1.num_rows(), 5)
        self.assertEqual(self.myPanelData.num_rows(), 5)


class TestVerifySegmentation(unittest.TestCase):
    def setUp(self):
        self.my_data = getData(1)

    def test_verify_segmentation_valid(self):
        # Define a valid segmentation
        valid_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping={1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3'}
        )

        # Verify segmentation
        self.assertIsNone(self.my_data.verify_segmentation(valid_segmentation))

    def test_verify_segmentation_invalid_variable(self):
        # Define a segmentation with an invalid variable
        invalid_variable_segmentation = DiscreteSegmentationTuple(
            variable='invalid_variable', mapping={1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3'}
        )

        # Verify segmentation and expect BiogemeError to be raised
        with self.assertRaises(BiogemeError):
            self.my_data.verify_segmentation(invalid_variable_segmentation)

    def test_verify_segmentation_missing_entries(self):
        # Define a segmentation with missing entries in the data
        missing_entries_segmentation = DiscreteSegmentationTuple(
            variable='Choice',
            mapping={
                1: 'Alt. 1',
                2: 'Alt. 2',
            },
        )

        # Verify segmentation and expect BiogemeError to be raised
        with self.assertRaises(BiogemeError) as cm:
            self.my_data.verify_segmentation(missing_entries_segmentation)
        self.assertIn('missing in the segmentation', str(cm.exception))

    def test_verify_segmentation_extra_entries(self):
        # Define a segmentation with extra entries not present in the data
        extra_entries_segmentation = DiscreteSegmentationTuple(
            variable='Choice',
            mapping={1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3', 4: 'Alt. 4'},
        )
        # Verify segmentation and expect BiogemeError to be raised
        with self.assertRaises(BiogemeError) as cm:
            self.my_data.verify_segmentation(extra_entries_segmentation)
        self.assertIn('do not exist in the data', str(cm.exception))


class TestGenerateSegmentation(unittest.TestCase):
    def setUp(self):
        self.my_data = getData(1)

    def test_generate_segmentation_valid(self):
        # Define a valid variable and mapping
        variable = 'Choice'
        mapping = {1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3'}

        # Generate segmentation
        segmentation = self.my_data.generate_segmentation(variable, mapping)

        expected_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=mapping
        )
        # Verify the generated segmentation
        self.assertIsInstance(segmentation, DiscreteSegmentationTuple)
        self.assertEqual(
            repr(segmentation.variable), repr(expected_segmentation.variable)
        )
        self.assertDictEqual(segmentation.mapping, expected_segmentation.mapping)

        segmentation = self.my_data.generate_segmentation(
            variable, mapping, reference='Alt. 2'
        )

        expected_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=mapping, reference='Alt. 2'
        )
        # Verify the generated segmentation
        self.assertIsInstance(segmentation, DiscreteSegmentationTuple)
        self.assertEqual(
            repr(segmentation.variable), repr(expected_segmentation.variable)
        )
        self.assertDictEqual(segmentation.mapping, expected_segmentation.mapping)
        self.assertEqual(segmentation.reference, expected_segmentation.reference)

        mapping = {1: 'Alt. 1', 2: 'Alt. 2'}
        expected_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 3: 'Choice_3'}

        # Generate segmentation
        segmentation = self.my_data.generate_segmentation(variable, mapping)

        expected_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=expected_mapping
        )
        # Verify the generated segmentation
        self.assertIsInstance(segmentation, DiscreteSegmentationTuple)
        self.assertEqual(
            repr(segmentation.variable), repr(expected_segmentation.variable)
        )
        self.assertDictEqual(segmentation.mapping, expected_segmentation.mapping)

    def test_generate_segmentation_invalid_reference(self):
        variable = 'Choice'
        mapping = {1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3'}

        # Generate segmentation
        with self.assertRaises(BiogemeError):
            _ = self.my_data.generate_segmentation(
                variable, mapping, reference='invalid_reference'
            )

    def test_generate_segmentation_invalid_variable(self):
        # Define an invalid variable
        invalid_variable = 'invalid_variable'
        mapping = {1: 'One', 2: 'Two'}

        # Generate segmentation and expect BiogemeError to be raised
        with self.assertRaises(BiogemeError):
            self.my_data.generate_segmentation(invalid_variable, mapping)

    def test_generate_segmentation_values_not_in_data(self):
        # Define a variable and mapping with values not present in the data
        variable = 'Choice'
        mapping = {1: 'One', 2: 'Two', 4: 'Four'}

        # Generate segmentation and expect BiogemeError to be raised
        with self.assertRaises(BiogemeError) as cm:
            self.my_data.generate_segmentation(variable, mapping)
        self.assertIn('do not exist in the data', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
