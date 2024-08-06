"""
Test the database module

:author: Michel Bierlaire
:data: Wed Apr 29 18:02:59 2020

"""

# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import biogeme.database as db
import biogeme.exceptions as excep
from biogeme.expressions import Variable, bioDraws, TypeOfElementaryExpression
from biogeme.native_draws import description_of_native_draws
from biogeme.segmentation import DiscreteSegmentationTuple
from test_data import (
    getData,
    output_flatten_database_1,
    output_flatten_database_3,
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
        with self.assertRaises(excep.BiogemeError):
            _ = db.Database('name', df)

    def test_bad_values(self):
        with self.assertRaises(excep.BiogemeError):
            df = getData(0)

    def test_valuesFromDatabase(self):
        expr = self.Variable1 + self.Variable2
        result = self.myData1.values_from_database(expr).tolist()
        self.assertListEqual(result, [11, 22, 33, 44, 55])
        # Check the exception when data is empty
        database = getData(1)
        database.data = database.data[0:0]
        with self.assertRaises(excep.BiogemeError):
            result = database.values_from_database(expr)

    def test_check_segmentation(self):
        the_data = getData(1)
        correct_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3'}
        correct_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=correct_mapping
        )
        output = the_data.check_segmentation(correct_segmentation)
        expected_output = {'Alt. 1': 2, 'Alt. 2': 2, 'Alt. 3': 1}
        self.assertDictEqual(output, expected_output)
        incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2'}
        incorrect_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=incorrect_mapping
        )
        with self.assertRaises(excep.BiogemeError):
            _ = the_data.check_segmentation(incorrect_segmentation)

        another_incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 4: 'Does not exist'}
        another_incorrect_segmentation = DiscreteSegmentationTuple(
            variable='Choice', mapping=another_incorrect_mapping
        )
        with self.assertRaises(excep.BiogemeError):
            _ = the_data.check_segmentation(another_incorrect_segmentation)

    def test_checkAvailabilityOfChosenAlt(self):
        avail = {1: self.Av1, 2: self.Av2, 3: self.Av3}
        result = self.myData1.check_availability_of_chosen_alt(
            avail, self.Choice
        ).tolist()
        self.assertListEqual(result, [False, True, True, True, True])
        # Check the exception when key is wrong
        with self.assertRaises(excep.BiogemeError):
            _ = self.myData1.check_availability_of_chosen_alt(
                {223: self.Av1}, self.Choice
            )

        # Check the exception when data is empty
        database = getData(1)
        database.data = database.data[0:0]
        with self.assertRaises(excep.BiogemeError):
            _ = database.check_availability_of_chosen_alt(avail, self.Choice)

    def test_choiceAvailabilityStatistics(self):
        avail = {1: self.Av1, 2: self.Av2, 3: self.Av3}
        result = self.myData1.choice_availability_statistics(avail, self.Choice)
        expected_result = {1: (2, 4), 2: (2, 5), 3: (1, 4)}
        self.assertDictEqual(result, expected_result)
        # Check the exception when data is empty
        database = getData(1)
        database.data = database.data[0:0]
        with self.assertRaises(excep.BiogemeError):
            _ = database.choice_availability_statistics(avail, self.Choice)

    def test_scale_column(self):
        self.myData1.scale_column('Variable1', 100)
        the_result = list(self.myData1.data['Variable1'])
        expected_result = [100, 200, 300, 400, 500]
        self.assertListEqual(the_result, expected_result)

    def test_suggest_scaling(self):
        result = self.myData1.suggest_scaling()
        result.reset_index(inplace=True, drop=True)
        expected_result = pd.DataFrame(
            {
                'Column': ['Variable2'],
                'Scale': [0.01],
                'Largest': [50],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result, check_index_type=False)
        result = self.myData1.suggest_scaling(columns=['Exclude', 'Variable2'])
        result.reset_index(inplace=True, drop=True)
        pd.testing.assert_frame_equal(result, expected_result, check_index_type=False)
        with self.assertRaises(excep.BiogemeError):
            _ = self.myData1.suggest_scaling(columns=['wrong_name'])

    def test_addColumn(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        expression = Variable2 * Variable1
        result = self.myData1.add_column(expression, 'NewVariable')
        theList = result.tolist()
        self.assertListEqual(theList, [10, 40, 90, 160, 250])
        with self.assertRaises(ValueError):
            result = self.myData1.add_column(expression, 'Variable1')
        database = getData(1)
        database.data = database.data[0:0]
        with self.assertRaises(excep.BiogemeError):
            result = database.add_column(expression, 'NewVariable')

    def test_DefineVariable(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        expression = Variable2 * Variable1
        result = self.myData1.define_variable('NewVariable', expression)
        theList = self.myData1.data['NewVariable'].tolist()
        self.assertListEqual(theList, [10, 40, 90, 160, 250])
        with self.assertRaises(ValueError):
            result = self.myData1.define_variable('Variable1', expression)
        database = getData(1)
        database.data = database.data[0:0]
        with self.assertRaises(excep.BiogemeError):
            result = database.define_variable('NewVariable', expression)

    def test_count(self):
        c = self.myData1.count('Person', 1)
        self.assertEqual(c, 3)

    def test_remove(self):
        Exclude = Variable('Exclude')
        d = getData(1)
        d.remove(Exclude)
        rows, _ = d.data.shape
        self.assertEqual(rows, 3)
        # This should not remove anything
        d.remove(0)
        rows, _ = d.data.shape
        self.assertEqual(rows, 3)

    def test_dumpOnFile(self):
        f = self.myData1.dump_on_file()
        exists = Path(f).is_file()
        os.remove(f)
        self.assertTrue(exists)

    def test_generateDraws(self):
        the_random_draws = description_of_native_draws()
        for draw_type in the_random_draws:
            random_draws = bioDraws('random_draws', draw_type)
            dict_of_draws = random_draws.dict_of_elementary_expression(
                the_type=TypeOfElementaryExpression.DRAWS
            )
            types = {name: draw.drawType for name, draw in dict_of_draws.items()}

            the_draws_table = self.myData1.generate_draws(types, ['random_draws'], 10)
            dim = the_draws_table.shape
            self.assertTupleEqual(dim, (5, 10, 1))

        with self.assertRaises(excep.BiogemeError):
            the_draws_table = self.myData1.generate_draws(
                {'random_draws': 'wrong_type'}, ['random_draws'], 10
            )

    def test_setRandomGenerators(self):
        def logNormalDraws(sampleSize, numberOfDraws):
            return np.exp(np.random.randn(sampleSize, numberOfDraws))

        def wrong_logNormalDraws(sampleSize, numberOfDraws):
            return np.exp(np.random.randn(2 * sampleSize, 2 * numberOfDraws))

        def exponentialDraws(sampleSize, numberOfDraws):
            return -1.0 * np.log(np.random.rand(sampleSize, numberOfDraws))

        # We associate these functions with a name
        theDict = {
            'LOGNORMAL': db.RandomNumberGeneratorTuple(
                generator=logNormalDraws,
                description='Draws from lognormal distribution',
            ),
            'EXP': db.RandomNumberGeneratorTuple(
                generator=exponentialDraws,
                description='Draws from exponential distributions',
            ),
            'WRONG_LOGNORMAL': db.RandomNumberGeneratorTuple(
                generator=wrong_logNormalDraws,
                description='Draws from lognormal distribution',
            ),
        }
        self.myData1.set_random_number_generators(theDict)

        # We can now generate draws from these distributions
        randomDraws1 = bioDraws('randomDraws1', 'LOGNORMAL')
        randomDraws2 = bioDraws('randomDraws2', 'EXP')
        x = randomDraws1 + randomDraws2
        dict_of_draws = x.dict_of_elementary_expression(
            the_type=TypeOfElementaryExpression.DRAWS
        )
        types = {
            name: expression.drawType for name, expression in dict_of_draws.items()
        }
        theDrawsTable = self.myData1.generate_draws(
            types, ['randomDraws1', 'randomDraws2'], 10
        )
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (5, 10, 2))

        with self.assertRaises(excep.BiogemeError):
            the_draws_table = self.myData1.generate_draws(
                {'random_draws': 'WRONG_LOGNORMAL'}, ['random_draws'], 10
            )

        with self.assertRaises(ValueError):
            self.myData1.set_random_number_generators(
                {'NORMAL': (logNormalDraws, 'Designed to generate an error')}
            )

    def test_sampleWithReplacement(self):
        res1 = self.myData1.sample_with_replacement()
        res2 = self.myData1.sample_with_replacement(12)
        dim1 = res1.shape
        dim2 = res2.shape
        self.assertTupleEqual(dim1, (5, 8))
        self.assertTupleEqual(dim2, (12, 8))

    def test_panel(self):
        # Data is not considered panel yet
        shouldBeFalse = self.myPanelData.is_panel()
        self.myPanelData.panel('Person')
        shouldBeTrue = self.myPanelData.is_panel()
        self.assertTrue(shouldBeTrue)
        self.assertFalse(shouldBeFalse)

        wrong_panel = getData(4)
        with self.assertRaises(excep.BiogemeError):
            wrong_panel.panel('Person')

    def test_panelDraws(self):
        randomDraws1 = bioDraws('randomDraws1', 'NORMAL')
        randomDraws2 = bioDraws('randomDraws2', 'UNIFORMSYM')
        # We build an expression that involves the two random variables
        x = randomDraws1 + randomDraws2
        dict_of_draws = x.dict_of_elementary_expression(
            the_type=TypeOfElementaryExpression.DRAWS
        )
        types = {
            name: expression.drawType for name, expression in dict_of_draws.items()
        }
        self.myPanelData.panel('Person')
        theDrawsTable = self.myPanelData.generate_draws(
            types, ['randomDraws1', 'randomDraws2'], 10
        )
        dim = theDrawsTable.shape
        self.assertTupleEqual(dim, (2, 10, 2))

    def test_generateFlatPanelDataframe(self):
        # If the data has not been declared panel, an exception is raised.
        with self.assertRaises(excep.BiogemeError):
            result = self.myPanelData.generate_flat_panel_dataframe()
        self.myPanelData.panel('Person')
        # Test with default parameters
        result1 = self.myPanelData.generate_flat_panel_dataframe()
        result1 = result1.reindex(sorted(result1.columns), axis='columns')
        compared1 = output_flatten_database_1.reindex(
            sorted(output_flatten_database_1.columns), axis='columns'
        )
        compared1.index.name = 'Person'
        pd.testing.assert_frame_equal(result1, compared1)
        # Test with automatic detection of identical columns
        result2 = self.myPanelData.generate_flat_panel_dataframe(identical_columns=None)
        result2 = result2.reindex(sorted(result2.columns), axis='columns')
        pd.testing.assert_frame_equal(result2, compared1)
        # Test with explicit list of identical columns
        result3 = self.myPanelData.generate_flat_panel_dataframe(
            identical_columns=['Age'], save_on_file=True
        )
        os.remove('test_3_flatten.csv')
        result3 = result3.reindex(sorted(result3.columns), axis='columns')
        compared3 = output_flatten_database_3.reindex(
            sorted(output_flatten_database_3.columns), axis='columns'
        )
        compared3.index.name = 'Person'
        pd.testing.assert_frame_equal(result3, compared3)

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
        self.assertEqual(self.myData1.get_number_of_observations(), 5)
        self.assertEqual(self.myPanelData.get_number_of_observations(), 5)

    def test_samplesSize(self):
        self.myPanelData.panel('Person')
        self.assertEqual(self.myData1.get_sample_size(), 5)
        self.assertEqual(self.myPanelData.get_sample_size(), 2)

    def test_sampleIndividualMapWithReplacement(self):
        self.myPanelData.panel('Person')
        res = self.myPanelData.sample_individual_map_with_replacement(10)
        dim = res.shape
        self.assertTupleEqual(dim, (10, 2))
        res = self.myPanelData.sample_individual_map_with_replacement(size=None)
        dim = res.shape
        self.assertTupleEqual(dim, (len(self.myPanelData.individualMap), 2))

        with self.assertRaises(excep.BiogemeError):
            _ = self.myData1.sample_individual_map_with_replacement(10)

    def test_split(self):
        data = {
            'userID': [
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
            ],
            'obsID': list(range(25)),
        }
        df = pd.DataFrame(data)
        database = db.Database('test', df)
        sp = database.split(5)
        estimation, validation = sp[0]
        # Verify that the intersection is empty
        intersection = set(estimation['obsID']) & set(validation['obsID'])
        self.assertSetEqual(intersection, set())
        # Verify that the union is the whole range
        union = set(estimation['obsID']) | set(validation['obsID'])
        self.assertSetEqual(union, set(range(25)))

        sp = database.split(5, groups='userID')
        estimation, validation = sp[0]
        # Verify that the intersection is empty
        intersection = set(estimation['userID']) & set(validation['userID'])
        self.assertSetEqual(intersection, set())
        # Verify that the union is the whole range
        union = set(estimation['userID']) | set(validation['userID'])
        self.assertSetEqual(union, set(range(1, 6)))

        with self.assertRaises(excep.BiogemeError):
            sp = database.split(1)

        self.myPanelData.panel('Person')
        with self.assertRaises(excep.BiogemeError):
            _ = self.myPanelData.split(2, groups='any_name')

        result = self.myPanelData.split(2)
        self.assertEqual(len(result), 2)


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
        with self.assertRaises(excep.BiogemeError):
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
        with self.assertRaises(excep.BiogemeError) as cm:
            self.my_data.verify_segmentation(missing_entries_segmentation)
        self.assertIn('missing in the segmentation', str(cm.exception))

    def test_verify_segmentation_extra_entries(self):
        # Define a segmentation with extra entries not present in the data
        extra_entries_segmentation = DiscreteSegmentationTuple(
            variable='Choice',
            mapping={1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3', 4: 'Alt. 4'},
        )
        # Verify segmentation and expect BiogemeError to be raised
        with self.assertRaises(excep.BiogemeError) as cm:
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
        with self.assertRaises(excep.BiogemeError):
            _ = self.my_data.generate_segmentation(
                variable, mapping, reference='invalid_reference'
            )

    def test_generate_segmentation_invalid_variable(self):
        # Define an invalid variable
        invalid_variable = 'invalid_variable'
        mapping = {1: 'One', 2: 'Two'}

        # Generate segmentation and expect BiogemeError to be raised
        with self.assertRaises(excep.BiogemeError):
            self.my_data.generate_segmentation(invalid_variable, mapping)

    def test_generate_segmentation_values_not_in_data(self):
        # Define a variable and mapping with values not present in the data
        variable = 'Choice'
        mapping = {1: 'One', 2: 'Two', 4: 'Four'}

        # Generate segmentation and expect BiogemeError to be raised
        with self.assertRaises(excep.BiogemeError) as cm:
            self.my_data.generate_segmentation(variable, mapping)
        self.assertIn('do not exist in the data', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
